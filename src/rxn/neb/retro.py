#!/usr/bin/env python

import pickle
import time
from copy import deepcopy
from typing import Any, Dict, List, Set

import numpy as np
from loguru import logger

from .fingerprints import load_fingerprints_model
from .single_step_predictions import load_single_step_prediction_model, run_rxn_prediction
from .utils.core import (
    Node,
    add_to_saved_reactions,
    check_reaction,
    check_sanity_tree,
    complete_reactions,
    compute_reaction_smiles_txt2fp_dict,
    fetch_score,
    filter_similar_node_id,
    filter_similar_node_id_with_tree,
    get_all_the_finished_reactions_up_to,
    get_full_text_sets_of_reactions,
    get_level_max,
    get_retrosynthesis,
    get_score_for_a_single_path,
    get_score_for_many_single_path,
    good_coverage,
    hash_str,
    initialize_tree,
    list_is_included,
    propagate_leaf,
    select_l_nb_filter_node_id,
    select_l_nb_node_id,
    smiles2fp,
    split_precursors_with_wait,
    update_reaction_smiles_from_tree,
)
from .utils.general import (
    check_alternatives_among,
    get_path_to_node,
    select_from_saved_here,
    select_prefix_to_keep,
)
from .utils.smiles import multistep_standardize

max_dist_along_cutoff = int(0.4 * 100)  # with PCA 16 dim is about 0.4 times the original


def retro_synthesis(
    input_data,
):
    logger.debug("retro_synthesis: neb-retro STARTING_retro_synthesis")
    logger.info("retro_synthesis: neb-retro")

    # input params from `input_data`:
    logger.debug("retro_synthesis: input_data[parameters] " + str(input_data["parameters"]))
    # those not available are set using defaults
    if "max_steps" in input_data["parameters"]:
        max_steps = input_data["parameters"]["max_steps"]
    else:
        max_steps = 12

    # Pruning the tree every N pruning_steps
    if "pruning_steps" in input_data["parameters"]:
        pruning_steps = input_data["parameters"]["pruning_steps"]
    else:
        pruning_steps = 2

    # `input_data` should contain also the path to the files needed by rxn-neb
    # Path to the pca_model_filename
    if "pca_model_filename" not in input_data:
        input_data["pca_model_filename"] = "./neb_pca_object_16.pkl"

    # Path to the tree_data_dict_pca_filename
    if "tree_data_dict_pca_filename" not in input_data:
        input_data["tree_data_dict_pca_filename"] = "./neb_tree_data_pca_rxnfps_16.pkl.bz2"

    product = input_data["product"]
    prefix = "rxn_neb_retro"
    # max_steps cannot be lower than 1
    if max_steps < 1:
        max_steps = 1
    logger.info(f"retro_synthesis: setting: max_steps: {max_steps}")
    clean = True
    # take_all = False
    # restart_from_knowledge = False  # this is now deprecated, on the UI not usable
    # product_can_exist = False  # Allow retro for an available product
    max_levels = 2 * max_steps
    min_levels = 4
    max_predictions = 40
    min_predictions = 6
    max_num_leaves_to_be_propagated_init = 4 + max_steps
    n_max_leaves_speed = 25
    n_max_leaves_speed_heavy_cut = 100
    ioniq = 1
    max_total_time = 2.0 * 60.0 * float(max_steps) + 240.0
    if max_steps > 12:
        max_total_time = 3.0 * 60.0 * float(max_steps) + 900.0
    max_step_time = 2.0 * 60.0 * float(max_steps)

    # Timing
    time0 = time.time()

    max_num_leaves_to_be_propagated = max_num_leaves_to_be_propagated_init

    if max_total_time < max_step_time:
        max_step_time = max_total_time
        logger.info("retro_synthesis: setting: max_step_time = max_total_time (seconds) " + str(max_total_time))

    logger.info(f"retro_synthesis: setting: max_total_time = {max_total_time} (seconds)")
    logger.info(f"retro_synthesis: setting: max_step_time = {max_step_time} (seconds)")
    logger.info(f"retro_synthesis: setting: max_steps = {max_steps} (from UI)")
    logger.info(f"retro_synthesis: setting: n_max_leaves_speed = {n_max_leaves_speed}")
    logger.info(f"retro_synthesis: setting: n_max_leaves_speed_heavy_cut = {n_max_leaves_speed_heavy_cut}")

    run_is_finished = False
    run_is_finished_with_status = "RUNNING"  #  Options are WAITING, RUNNING, DONE, ERROR

    retrosynthesis: Dict[str, Any] = {}
    retrosynthesis["status"] = run_is_finished_with_status
    retrosynthesis["routes"] = []

    # data_rxnfps = np.load('pca_rxnfps_16_transform.npy')
    # data_rxnfps = np.load('simpler_pca_rxnfps_16_transform.npy')
    # PCA object to embed new predicted steps
    with open(input_data["pca_model_filename"], "rb") as f_pca:
        # e.g. "neb_pca_object_16.pkl"
        pca_model = pickle.load(f_pca)
    logger.info("retro_synthesis: Loading neb_pca_object_16.pkl")

    # Reaction fingerprints of dataset in PCA space
    # with bz2.open(
    with open(
        # e.g. "neb_tree_data_pca_rxnfps_16_leaf_size_8_linear_plus_tree.pkl.bz2"
        input_data["tree_data_dict_pca_filename"],
        "rb",
    ) as f:
        tree_data_dict_pca = pickle.load(f)
    logger.info("retro_synthesis: Loading tree_data_dict_pca_filename")

    logger.info("retro_synthesis: starting...")
    logger.info("retro_synthesis: target product: " + str(product))
    # product = multistep_standardize(product, model_settings=model_settings)
    product = multistep_standardize(product)
    logger.info("retro_synthesis: target product: " + str(product) + " (after canonical)")

    my_exclude = set()
    # add the product
    my_exclude.add(product)

    # Loading models
    fw_model_path = input_data["forward_model_path"]
    re_model_path = input_data["backward_model_path"]
    fw_model = load_single_step_prediction_model(model_path=fw_model_path, n_best=1, beam_size=10, max_length=300)
    re_model = load_single_step_prediction_model(model_path=re_model_path, topn=10, n_best=10, beam_size=15, max_length=300)

    fingerprints_model_path = input_data["fingerprints_model_path"]
    fingerprints_model = load_fingerprints_model(model_path=fingerprints_model_path)

    # This is the dictionary to save the confidences of every reaction smile
    saved_classes: Dict[str, str] = {}
    saved_confidences: Dict[str, float] = {}
    saved_reactions: Dict[str, Set[str]] = {}

    logger.info("retro_synthesis: product_via_run_rxn_prediction " + str(product))
    retro_predictions_good = run_rxn_prediction(product, fw_model=fw_model, re_model=re_model)

    # This may include very similar reactions ... like `duplicates` or
    #    trivial modifications.
    saved_reactions = add_to_saved_reactions(product, retro_predictions_good, saved_reactions)
    logger.info("len(retro_predictions_good) before cleaning: " + str(len(retro_predictions_good)))
    if clean and len(retro_predictions_good) > 0:
        # Sort the list of strings by string len
        sorted_retro_predictions_good = sorted(retro_predictions_good, key=lambda el: len(el))

        # Filter the "trivial" duplicates
        r_smiles = [r + ">>" + product for r in sorted_retro_predictions_good]
        fp = smiles2fp(r_smiles, pca_model, fingerprints_model=fingerprints_model)
        keep = []
        keep.append(r_smiles[0])
        n = len(r_smiles)
        for i in range(1, n):
            r = r_smiles[i]
            if check_reaction(keep, r, r_smiles, fp):
                # print('keep',i)
                keep.append(r)

        retro_predictions_good = [r.split(">>")[0] for r in keep]

    logger.info("len(retro_predictions_good) after cleaning: " + str(len(retro_predictions_good)))

    tree_dict = initialize_tree(product, retro_predictions_good)

    # First initialization
    full_node_list_from_tree = [tree_dict[node].name for node in tree_dict]

    # Finds lever after level, all the reaction nodes and
    #   create a list of reaction_smiles stored in `reaction_smiles_txt`
    #   this is appended. More over it creates a mapping node to reaction_smile
    #   2 outputs, 3 inputs

    # Initialization
    reaction_smiles_txt = []
    map_node2react_smile: Dict[str, str] = {}
    node_reaction_list: List[str] = []
    reaction_smiles_txt2fp_dict: Dict[str, Any] = {}
    already_finished: List[List[int]] = []
    saved_scores: Dict[str, Any] = {
        "global": {},
        "local": {},
        "part": {},
        "orig_part": {},
        "part_inds": {},
        "orig_part_inds": {},
        "part_len_steps": {},
        "part_steps": {},
        "orig_part_dists": {},
    }
    results_v_l_score = []
    results_v_l_score_dict = {}
    n_finished = 0
    node_id_terminated: Set[str] = set()
    sources: List[Dict[str, Any]] = []

    # Extract for the first time the reaction smiles from tree
    map_node2react_smile, node_reaction_list = update_reaction_smiles_from_tree(tree_dict, map_node2react_smile, node_reaction_list)

    # save also in the txt list file
    reaction_smiles_txt = [map_node2react_smile[s] for s in node_reaction_list]

    reaction_smiles_txt2fp_dict = compute_reaction_smiles_txt2fp_dict(
        reaction_smiles_txt2fp_dict,
        reaction_smiles_txt,
        pca_model,
        fingerprints_model=fingerprints_model,
    )

    reaction_smiles_fp = np.array([reaction_smiles_txt2fp_dict[item] for item in reaction_smiles_txt])

    logger.info("retro_synthesis: l_select: 0 before")
    logger.info("retro_synthesis: len(map_node2react_smile): " + str(len(node_reaction_list)))
    logger.info("retro_synthesis: len(reaction_smiles_txt2fp_dict): " + str(len(reaction_smiles_txt2fp_dict)))
    logger.info("retro_synthesis: len(reaction_smiles_txt): " + str(len(reaction_smiles_txt)))
    logger.info("retro_synthesis: len(tree_dict): " + str(len(tree_dict)))
    logger.info("retro_synthesis: reaction_smiles_fp.shape: " + str(reaction_smiles_fp.shape))

    # list_uniq_all_level = {}

    l_select = 1

    list_of_list_of_1_inds = select_l_nb_node_id(tree_dict, node_reaction_list, l_select)
    # Correction to remove empty lists [[],[],[]...]
    list_of_list_of_1_inds = [l1 for l1 in list_of_list_of_1_inds if len(l1) > 0]
    list_of_list_of_1_inds_ = [[0, l1[0]] for l1 in list_of_list_of_1_inds if len(l1) > 0]
    list_uniq_arti = filter_similar_node_id(
        list_of_list_of_1_inds_,
        node_reaction_list,
        map_node2react_smile,
        reaction_smiles_txt2fp_dict,
    )
    list_uniq = [[l1[1]] for l1 in list_uniq_arti]
    # list_uniq_all_level[1] = [l1 for l1 in list_uniq]

    logger.info(f"retro_synthesis: list_uniq at level L=1: {list_uniq}")
    n_opena_at_l1 = len(list_uniq)  # this is used in the coverage

    # Check until L=2
    level = 2
    finished_dict, _ = get_all_the_finished_reactions_up_to(tree_dict, node_reaction_list, level)
    logger.debug(f"retro_synthesis: finished_dict_DEBUG {len(finished_dict)} {finished_dict}")
    logger.debug(f"retro_synthesis: node_reaction_list_DEBUG {len(node_reaction_list)}")
    logger.debug(f"retro_synthesis: tree_dict_DEBUG {len(tree_dict)}")

    # Print finished reactions (if present)
    logger.info("retro_synthesis: List of finished reactions:")
    for _, v in finished_dict.items():
        if len(v) > 0:
            for v_ in v:
                if v_ in already_finished:
                    logger.info("retro_synthesis: already_finished " + str(v_))
                else:
                    logger.info("retro_synthesis: just_finished " + str(v_) + " including level: " + str(level))
                    already_finished.append(v_)
                    saved_scores = get_score_for_a_single_path(
                        v_,
                        tree_data_dict_pca,
                        node_reaction_list,
                        map_node2react_smile,
                        reaction_smiles_txt2fp_dict,
                        saved_scores,
                        max_dist_along_cutoff,
                    )

                    single_result_v_l_score = fetch_score(
                        v_,
                        saved_scores,
                        node_reaction_list,
                        tree_dict,
                        full_node_list_from_tree,
                        map_node2react_smile,
                        True,
                    )
                    results_v_l_score.append(single_result_v_l_score)

                    l_str = single_result_v_l_score["l_str"]
                    results_v_l_score_dict[l_str] = single_result_v_l_score
                    n_finished = n_finished + 1
                    logger.info("retro_synthesis: Check_n_finished_vs_len(results_v_l_score_dict)_posA:")
                    logger.info("retro_synthesis: " + str(n_finished) + " " + str(len(results_v_l_score_dict)))

                    logger.info("retro_synthesis: Aggregate: save final_score, single step synthesis")
                    length = len(single_result_v_l_score["v"])  # len() = 1
                    score = results_v_l_score_dict[l_str]["score"]
                    score_initial = score * length
                    results_v_l_score_dict[l_str]["final_score"] = [
                        score,
                    ]

                    # list of list
                    results_v_l_score_dict[l_str]["complete_reactions"] = [[]]
                    logger.info("retro_synthesis: Aggregated_final_score l_str/final_score/complete_reactions")
                    logger.info("retro_synthesis: " + str(results_v_l_score_dict[l_str]["final_score"]))
                    logger.info("retro_synthesis: " + str(results_v_l_score_dict[l_str]["v"]))
                    logger.info("retro_synthesis: " + str(results_v_l_score_dict[l_str]["l"]))
                    logger.info("retro_synthesis: " + str(results_v_l_score_dict[l_str]["complete_reactions"]))
                    logger.info("retro_synthesis ---")

    logger.info("retro_synthesis: Assemble full smiles reactions n_finished = " + str(n_finished))
    for k_, v_ in results_v_l_score_dict.items():
        # Here the if is not necessary, we passed only once
        #   later it may be useful, but a priori longer path
        #   can finish and open new combinations ...
        # if not 'full_smiles' in results_v_l_score_dict:
        results_v_l_score_dict[k_]["full_smiles"] = get_full_text_sets_of_reactions(v_, results_v_l_score_dict)

        results_v_l_score_dict[k_]["source"] = [hash_str(str(item)) for item in results_v_l_score_dict[k_]["full_smiles"]]

    l_str_present = set([item["l_str"] for item in sources])
    for k_, v_ in results_v_l_score_dict.items():
        for i in range(len(results_v_l_score_dict[k_]["source"])):
            if results_v_l_score_dict[k_]["l_str"] in l_str_present:
                continue
            dict_ = {}
            dict_["source"] = results_v_l_score_dict[k_]["source"][i]
            dict_["final_score"] = results_v_l_score_dict[k_]["final_score"][i]
            dict_["l_str"] = results_v_l_score_dict[k_]["l_str"]
            message_ = "retro_synthesis: SOURCE: " + str(dict_["source"]) + " " + str(dict_["final_score"]) + " " + str(dict_["l_str"])
            logger.info(message_)
            sources.append(dict_)

    logger.info("retro_synthesis: Check_n_finished_vs_len(results_v_l_score_dict)_posB: " + str(n_finished) + " " + str(len(results_v_l_score_dict)))

    logger.info(
        "retro_synthesis: results_v_l_score len(): " + str(len(results_v_l_score)) + " " + "results_v_l_score.pkl at level: " + str(level) + " " + "0"
    )

    logger.info("retro_synthesis: saved_scores: " + str(len(saved_scores)) + " saved_scores.pkl at level: " + str(level) + " 0")

    if level >= max_levels:
        run_is_finished = True

    time2 = time.time()
    message_ = "retro_synthesis: TIMING_LOOP_STEP " + str(time2 - time0) + " level: " + str(level) + " " + "n_finished: " + str(n_finished)
    logger.info(message_)

    skip_set: Set[str] = set()
    leaves_purged_set: Set[str] = set()

    # NOTE: LOOP step
    for step in range(100):
        time1 = time.time()
        logger.info(f"retro_synthesis: step: {step}")

        if level >= max_levels:
            run_is_finished = True

        if run_is_finished:
            logger.info("retro_synthesis: run_is_finished " + str(run_is_finished) + " current level " + str(level))
            break

        # Compounds l_select = 2,4,6,.. even numbers
        l_select = l_select + 1

        sanity_status = check_sanity_tree(tree_dict, leaves_purged_set)

        if sanity_status["check"]:
            # skip_set = set()
            for name_ in sanity_status["list_of_node_more_to_stop"]:
                skip_set.add(name_)
                parent_ = tree_dict[name_].parent
                name_c = name_.replace("_MORE", "_STOP")
                tree_dict[name_c] = Node(name_c, parent=parent_)

        logger.info("retro_synthesis: INFO_check_sanity_tree_len(skip_set): " + str(len(skip_set)))

        # Since the tree_dict may have been updated
        #  do update also the node_list
        if not len(full_node_list_from_tree) == len(tree_dict):
            for node in tree_dict:
                if node not in full_node_list_from_tree:
                    full_node_list_from_tree.append(node)

        leaves_to_be_propagated = []
        for node in tree_dict:
            if "MORE___" in node:
                if not tree_dict[node].depth == l_select:
                    continue
                # if node in skip_set:
                #    print(
                #        "list_of_node_more_to_stop_leaves_to_be_propagated_SKIP", node
                #    )
                #    continue
                if node in leaves_purged_set:
                    logger.info(f"retro_synthesis: LEAF_IN_leaves_purged_set_SKIP {node}")
                    continue
                if node not in sanity_status["more_status"]:
                    continue
                if sanity_status["more_status"][node] == "BAD":
                    continue
                message_ = "retro_synthesis: LEAF_TO_BE_PROPAGATED l_select:"
                message_ = message_ + " " + str(l_select)
                message_ = message_ + " " + str(tree_dict[node].depth)
                message_ = message_ + " NODE: " + str(node)
                logger.info(message_)
                leaves_to_be_propagated.append(node)

        logger.info(
            f"retro_synthesis: LEAVES_TO_BE_PROPAGATED_ORIG: l_select = {l_select}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}"
        )

        if len(leaves_to_be_propagated) < 1:
            # There is nothing to be propagated, the run is finished, exit the loop
            logger.info(f"retro_synthesis: finished: no leaves_to_be_propagated {len(leaves_to_be_propagated)}")
            run_is_finished = True
            logger.info("retro_synthesis: run_is_finished {run_is_finished} current level {level}")
            logger.debug("retro_synthesis: run_is_finished {run_is_finished} current level {level}")
            break

        # Here we filter the tree...
        if step % pruning_steps == 0 or len(leaves_to_be_propagated) > n_max_leaves_speed:
            max_num_leaves_to_be_propagated = max_num_leaves_to_be_propagated_init
            hard_reset = False

            # Backup leaves_to_be_propagated
            leaves_to_be_propagated_bak = deepcopy(leaves_to_be_propagated)
            logger.info(f"retro_synthesis: pruning: here we filter the tree: step = {step}, l_select = {l_select}")

            heavy_pruning = False
            try:
                n_pgr = len(leaves_to_be_propagated)
                if n_pgr > n_max_leaves_speed_heavy_cut:
                    heavy_pruning = True

                    if not pruning_steps == 1:
                        logger.info(f"retro_synthesis: setting pruning_steps = 1 from {pruning_steps}")
                        pruning_steps = 1

                    l_list_of_leaves = []
                    l_list_of_leaves_bak = []
                    for leaf_ in leaves_to_be_propagated:
                        li = [node for node in get_path_to_node(tree_dict[leaf_])]
                        l_list_of_leaves.append([full_node_list_from_tree.index(i) for i in li])
                        l_list_of_leaves_bak.append([full_node_list_from_tree.index(i) for i in li])

                    keep_index: List[int] = []
                    logger.info(f"retro_synthesis: HEAVY PRUNING: level = {level}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}")

                    l_list_single = l_list_of_leaves[0]
                    for imytry in range(0, 10):
                        ipos_limit = len(l_list_single) - 2 * imytry
                        if ipos_limit < 1:
                            break
                        logger.info(f"retro_synthesis: HEAVY PRUNING: ipos_limit = {ipos_limit}, attempt = {imytry}, len() = {len(l_list_single)}")

                        l_list_keep = []
                        keep_index = []
                        n_rejecting = 0
                        l_list_keep.append(l_list_of_leaves[0])
                        keep_index.append(0)
                        for il1_ in range(1, len(l_list_of_leaves)):
                            l_list_single = l_list_of_leaves[il1_]
                            # logger.info(
                            #     f"retro_synthesis: HEAVY PRUNING: trying {l_list_single}"
                            # )
                            take_, _ = check_alternatives_among(ipos_limit, l_list_single, l_list_keep)
                            if take_:
                                l_list_keep.append(l_list_single)
                                keep_index.append(il1_)
                                # logger.info(
                                #     f"retro_synthesis: HEAVY PRUNING: adding {l_list_single}"
                                # )
                            else:
                                n_rejecting = n_rejecting + 1
                                pass
                                # logger.info(
                                #     f"retro_synthesis: HEAVY PRUNING: n_sim = {n_sim} rejecting {l_list_single}"
                                # )
                        logger.info(f"retro_synthesis: HEAVY PRUNING: attempt {imytry} tot num. {n_pgr} keeping {len(l_list_keep)}")
                        logger.info(f"retro_synthesis: HEAVY PRUNING: attempt {imytry} n_rejecting = {n_rejecting}")
                        if len(l_list_keep) < n_max_leaves_speed_heavy_cut:
                            logger.info(
                                f"retro_synthesis: HEAVY PRUNING: attempt {imytry} len(l_list_keep) {len(l_list_keep)} < {n_max_leaves_speed_heavy_cut}"
                            )
                            break
                        l_list_of_leaves = [item for item in l_list_keep]
                        logger.info(f"retro_synthesis: HEAVY PRUNING: attempt {imytry} not enough {len(l_list_keep)}")
                    if heavy_pruning:
                        keep_index = [l_list_of_leaves_bak.index(item) for item in l_list_keep]
                        leaves_to_be_propagated = [leaves_to_be_propagated_bak[i] for i in keep_index]
                        # update backup
                        leaves_to_be_propagated_bak = deepcopy(leaves_to_be_propagated)
            except Exception:
                leaves_to_be_propagated = deepcopy(leaves_to_be_propagated_bak)
                logger.debug("retro_synthesis: HEAVY PRUNING FAILED WITH Exception")
                logger.info("retro_synthesis: HEAVY PRUNING FAILED WITH Exception")

            # compute scores ...
            logger.info(
                f"retro_synthesis: before PRUNING len(leaves_to_be_propagated) {len(leaves_to_be_propagated)} heavy_pruning = {heavy_pruning}"
            )
            v_list_of_leaves = []
            for leaf_ in leaves_to_be_propagated:
                vi = [node for node in get_path_to_node(tree_dict[leaf_])][1::2]
                v_list_of_leaves.append([node_reaction_list.index(i) for i in vi])

            saved_scores, saved_here = get_score_for_many_single_path(
                v_list_of_leaves,
                tree_data_dict_pca,
                node_reaction_list,
                map_node2react_smile,
                reaction_smiles_txt2fp_dict,
                saved_scores,
                max_dist_along_cutoff,
            )

            try:
                itry = 0
                while True:
                    # if heavy_pruning:
                    #     logger.info(
                    #         f"retro_synthesis: skip PRUNING because of heavy_pruning = {heavy_pruning}"
                    #     )
                    #     break
                    logger.info(
                        f"retro_synthesis: entering PRUNING len(leaves_to_be_propagated) {len(leaves_to_be_propagated)} heavy_pruning = {heavy_pruning}"
                    )
                    v_list_of_leaves = []
                    for leaf_ in leaves_to_be_propagated:
                        vi = [node for node in get_path_to_node(tree_dict[leaf_])][1::2]
                        v_list_of_leaves.append([node_reaction_list.index(i) for i in vi])

                    saved_scores, saved_here = get_score_for_many_single_path(
                        v_list_of_leaves,
                        tree_data_dict_pca,
                        node_reaction_list,
                        map_node2react_smile,
                        reaction_smiles_txt2fp_dict,
                        saved_scores,
                        max_dist_along_cutoff,
                    )

                    if step == 0:
                        ioniq = 4  # at the beginning we filter less
                    elif step == 1:
                        ioniq = 2
                    else:
                        ioniq = 1

                    list_best = select_from_saved_here(saved_here, max_num_leaves_to_be_propagated * ioniq)
                    logger.info(f"retro_synthesis: max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}, multi: {ioniq}")
                    new_leaves_to_be_propagated = []
                    for i__ in range(len(list_best)):
                        item = list_best[i__]
                        ipos = [i for i in range(len(v_list_of_leaves)) if v_list_of_leaves[i] == item]
                        for i in ipos:
                            leaf_ = leaves_to_be_propagated[i]
                            message_ = "retro_synthesis: LEAF_TO_BE_PROPAGATED_AFTER: "
                            message_ = message_ + "level "
                            message_ = message_ + str(level) + " "
                            message_ = message_ + str(i) + " "
                            message_ = message_ + str(tree_dict[leaf_].depth) + " "
                            message_ = message_ + "NODE: " + str(leaf_)
                            logger.info(message_)
                            if leaf_ not in new_leaves_to_be_propagated:
                                new_leaves_to_be_propagated.append(leaf_)
                            if leaf_ in sanity_status["to_complete"]:
                                leaves_to_complete = sanity_status["to_complete"][leaf_]
                                for le in leaves_to_complete:
                                    logger.info("retro_synthesis: LEAF_TO_COMPLETE: " + str(tree_dict[le].depth) + " NODE: " + str(le))
                                    if le not in new_leaves_to_be_propagated:
                                        new_leaves_to_be_propagated.append(le)

                    leaves_purged = [i for i in leaves_to_be_propagated if i not in new_leaves_to_be_propagated]

                    # Update the accumulated set of nodes that should not be used
                    leaves_purged_set = leaves_purged_set.union(set(leaves_purged))

                    leaves_to_be_propagated = [i for i in new_leaves_to_be_propagated]

                    logger.info(
                        f"""retro_synthesis: LOOP_LEAVES_TO_BE_PROPAGATED_AFTER_PURGED: loop: {itry}, level = {level}, l_select = {l_select}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}, len(leaves_purged_set) = {len(leaves_purged_set)}, max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}, multi: {ioniq}"""
                    )

                    if len(leaves_to_be_propagated) <= n_max_leaves_speed:
                        logger.info(
                            f"retro_synthesis: len(leaves_to_be_propagated) {len(leaves_to_be_propagated)} vs. n_max_leaves_speed: {n_max_leaves_speed}"
                        )
                        break

                    ioniq = 1
                    if hard_reset:
                        pass
                    elif max_num_leaves_to_be_propagated == 3:
                        max_num_leaves_to_be_propagated = 2
                        logger.info(f"retro_synthesis: max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}")
                        continue
                    elif max_num_leaves_to_be_propagated == 2:
                        max_num_leaves_to_be_propagated = 1
                        logger.info(f"retro_synthesis: max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}")
                        hard_reset = True
                        continue
                    else:
                        max_num_leaves_to_be_propagated = max_num_leaves_to_be_propagated - 3
                        if max_num_leaves_to_be_propagated <= 1:
                            max_num_leaves_to_be_propagated = 1
                            hard_reset = True
                            logger.info(
                                f"retro_synthesis: purge: hard_reset: {hard_reset}"
                                f"retro_synthesis: max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}"
                            )
                            continue

                    itry = itry + 1
                    if itry > 5:
                        break

                logger.info(
                    f"""retro_synthesis: LEAVES_TO_BE_PROPAGATED_AFTER_PURGED: level = {level}, l_select = {l_select}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}, len(leaves_purged_set) = {len(leaves_purged_set)}, max_num_leaves_to_be_propagated: {max_num_leaves_to_be_propagated}, multi: {ioniq}, hard_reset: {hard_reset}"""
                )

                # Tree is purged, set original value `max_num_leaves_to_be_propagated`
                max_num_leaves_to_be_propagated = max_num_leaves_to_be_propagated_init

            except Exception:
                leaves_to_be_propagated = deepcopy(leaves_to_be_propagated_bak)
                logger.debug(
                    f"retro_synthesis: PRUNING Exception using leaves_to_be_propagated_bak len(leaves_to_be_propagated_bak) {len(leaves_to_be_propagated_bak)}"
                )
                logger.info(
                    f"retro_synthesis: PRUNING Exception using leaves_to_be_propagated_bak len(leaves_to_be_propagated_bak) {len(leaves_to_be_propagated_bak)}"
                )

        # 2nd pass of heavy pruning: `2ND_HEAVY PRUNING`
        leaves_to_be_propagated_bak = deepcopy(leaves_to_be_propagated)
        heavy_pruning = False
        try:
            n_pgr = len(leaves_to_be_propagated)
            if n_pgr > n_max_leaves_speed_heavy_cut:
                heavy_pruning = True
                logger.info(f"retro_synthesis: entering 2ND_HEAVY PRUNING n_pgr = {n_pgr}")

                l_list_of_leaves = []
                l_list_of_leaves_bak = []
                for leaf_ in leaves_to_be_propagated:
                    li = [node for node in get_path_to_node(tree_dict[leaf_])]
                    l_list_of_leaves.append([full_node_list_from_tree.index(i) for i in li])
                    l_list_of_leaves_bak.append([full_node_list_from_tree.index(i) for i in li])

                keep_index = []
                logger.info(f"retro_synthesis: 2ND_HEAVY PRUNING: level = {level}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}")

                l_list_single = l_list_of_leaves[0]
                for imytry in range(0, 10):
                    ipos_limit = len(l_list_single) - 2 * imytry
                    if ipos_limit < 1:
                        break
                    logger.info(f"retro_synthesis: HEAVY PRUNING: ipos_limit = {ipos_limit}, attempt = {imytry}, len() = {len(l_list_single)}")

                    l_list_keep = []
                    keep_index = []
                    n_rejecting = 0
                    l_list_keep.append(l_list_of_leaves[0])
                    keep_index.append(0)
                    for il1_ in range(1, len(l_list_of_leaves)):
                        l_list_single = l_list_of_leaves[il1_]
                        take_, _ = check_alternatives_among(ipos_limit, l_list_single, l_list_keep)
                        if take_:
                            l_list_keep.append(l_list_single)
                            keep_index.append(il1_)
                        else:
                            n_rejecting = n_rejecting + 1
                            pass
                    logger.info(f"retro_synthesis: 2ND_HEAVY PRUNING: attempt {imytry} tot num. {n_pgr} keeping {len(l_list_keep)}")
                    logger.info(f"retro_synthesis: 2ND_HEAVY PRUNING: attempt {imytry} n_rejecting = {n_rejecting}")
                    if len(l_list_keep) < n_max_leaves_speed_heavy_cut:
                        logger.info(
                            f"retro_synthesis: 2ND_HEAVY PRUNING: attempt {imytry} len(l_list_keep) {len(l_list_keep)} < {n_max_leaves_speed_heavy_cut}"
                        )
                        break
                    l_list_of_leaves = [item for item in l_list_keep]
                    logger.info(f"retro_synthesis: 2ND_HEAVY PRUNING: attempt {imytry} not enough {len(l_list_keep)}")
                if heavy_pruning:
                    keep_index = [l_list_of_leaves_bak.index(item) for item in l_list_keep]
                    leaves_to_be_propagated = [leaves_to_be_propagated_bak[i] for i in keep_index]
                    # update backup
                    leaves_to_be_propagated_bak = deepcopy(leaves_to_be_propagated)
                    logger.info(
                        f"""retro_synthesis: LEAVES_TO_BE_PROPAGATED_AFTER_PURGED_and_2ND_HEAVY PRUNING: level = {level}, len(leaves_to_be_propagated) = {len(leaves_to_be_propagated)}"""
                    )

            # Tree is purged, set original value `max_num_leaves_to_be_propagated`
        except Exception:
            leaves_to_be_propagated = deepcopy(leaves_to_be_propagated_bak)
            logger.debug("retro_synthesis: 2ND_HEAVY PRUNING FAILED WITH Exception")
            logger.info("retro_synthesis: 2ND_HEAVY PRUNING FAILED WITH Exception")
        # end of 2nd pass of heavy pruning

        # This option in the last resort, with the 2nd_heavy pruning,
        # it should seldom pass in the following if statement
        if len(leaves_to_be_propagated) > n_max_leaves_speed_heavy_cut:
            leaves_to_be_propagated = deepcopy(leaves_to_be_propagated_bak)
            logger.info("retro_synthesis: call select_prefix_to_keep: " + str(len(leaves_to_be_propagated)))
            prefix = select_prefix_to_keep(node_reaction_list, saved_here)
            logger.info(f"retro_synthesis: call select_prefix_to_keep: prefix: {prefix}")
            new_list = [s for s in leaves_to_be_propagated if s.startswith(prefix)]
            logger.info(f"retro_synthesis: call select_prefix_to_keep: len(new_list): {len(new_list)}")
            if len(new_list) > 0:
                leaves_to_be_propagated = [item for item in new_list]
                logger.info("retro_synthesis: call select_prefix_to_keep: " + str(len(leaves_to_be_propagated)))
            else:
                leaves_to_be_propagated = deepcopy(leaves_to_be_propagated_bak)
                logger.info("retro_synthesis: call select_prefix_to_keep: KEEP ALL")
        # end of the select prefix section

        for leaf in leaves_to_be_propagated:
            (
                tree_dict,
                saved_reactions,
                saved_confidences,
                saved_classes,
            ) = propagate_leaf(
                leaf,
                tree_dict,
                saved_reactions,
                clean,
                pca_model,
                saved_confidences,
                saved_classes,
                fw_model=fw_model,
                re_model=re_model,
                fingerprints_model=fingerprints_model,
            )

            time2 = time.time()
            if n_finished > 0 and (time2 - time0) > max_step_time:
                run_is_finished = True
                logger.info(
                    f"""retro_synthesis: TIMING_TMP_in_leaves_to_be_propagated_step_time_END {time2 - time0} max_step_time: {max_step_time}, n_finished: {n_finished}"""
                )
                logger.info(f"retro_synthesis: WALLTIME_run_is_finished: {run_is_finished}, n_finished: {n_finished}, level: {level}, time2: {time2}")
                break

        if not len(full_node_list_from_tree) == len(tree_dict):
            for node in tree_dict:
                if node not in full_node_list_from_tree:
                    full_node_list_from_tree.append(node)

        level = l_select + 1  # `level` here is an odd number 3,5,7,...

        # operations after updating `tree_dict`: extract the reaction smiles from tree
        map_node2react_smile, node_reaction_list = update_reaction_smiles_from_tree(tree_dict, map_node2react_smile, node_reaction_list)

        # save also in the txt list file
        reaction_smiles_txt = [map_node2react_smile[s] for s in node_reaction_list]
        reaction_smiles_txt2fp_dict = compute_reaction_smiles_txt2fp_dict(
            reaction_smiles_txt2fp_dict,
            reaction_smiles_txt,
            pca_model,
            fingerprints_model=fingerprints_model,
        )
        reaction_smiles_fp = np.array([reaction_smiles_txt2fp_dict[item] for item in reaction_smiles_txt])

        # We have level L=3,5,7,...
        list_of_list_of_l_inds = select_l_nb_filter_node_id(tree_dict, node_reaction_list, level, list_uniq)
        list_of_list_of_l_inds = [l1 for l1 in list_of_list_of_l_inds if len(l1) > 0]

        # list_uniq = [l1 for l1 in list_of_list_of_l_inds if len(l1) > 0]

        list_uniq = filter_similar_node_id_with_tree(
            tree_dict,
            list_of_list_of_l_inds,
            node_reaction_list,
            map_node2react_smile,
            reaction_smiles_txt2fp_dict,
        )

        if run_is_finished:
            logger.info(f"retro_synthesis: run_is_finished {run_is_finished} current level {level}")
            break

        # init `node_to_continue`
        node_to_continue = set([item[-1] for item in list_uniq])
        node_id_to_continue = set([node_reaction_list[i] for i in node_to_continue])
        leaves_to_be_propagated = [node for node in node_id_to_continue]
        l_select = l_select + 1
        logger.info(f"retro_synthesis: l_select {l_select} {len(leaves_to_be_propagated)}")

        tree_dict = split_precursors_with_wait(tree_dict, leaves_to_be_propagated, node_id_terminated, my_exclude)

        if not len(full_node_list_from_tree) == len(tree_dict):
            for node in tree_dict:
                if node not in full_node_list_from_tree:
                    full_node_list_from_tree.append(node)
        level = l_select + 1
        finished_dict, _ = get_all_the_finished_reactions_up_to(tree_dict, node_reaction_list, level)

        # Append - just in case... to make sure order of existing is not changed
        # full_node_list_from_tree=[tree_dict[node].name for node in tree_dict]
        for node in tree_dict:
            if node not in full_node_list_from_tree:
                full_node_list_from_tree.append(node)

        # Print finished reactions (if present)
        logger.info("retro_synthesis: List of finished reactions:")
        just_finished_are = []
        for k, v in finished_dict.items():
            if len(v) > 0:
                for v_ in v:
                    if v_ in already_finished:
                        logger.info("retro_synthesis: already_finished " + str(v_))
                    else:
                        logger.info("retro_synthesis: just_finished " + str(v_) + " including level: " + str(level))
                        already_finished.append(v_)
                        saved_scores = get_score_for_a_single_path(
                            v_,
                            tree_data_dict_pca,
                            node_reaction_list,
                            map_node2react_smile,
                            reaction_smiles_txt2fp_dict,
                            saved_scores,
                            max_dist_along_cutoff,
                        )

                        # print the score for the `just_finished` path
                        single_result_v_l_score = fetch_score(
                            v_,
                            saved_scores,
                            node_reaction_list,
                            tree_dict,
                            full_node_list_from_tree,
                            map_node2react_smile,
                            True,
                        )
                        results_v_l_score.append(single_result_v_l_score)

                        l_str = single_result_v_l_score["l_str"]
                        results_v_l_score_dict[l_str] = single_result_v_l_score
                        n_finished = n_finished + 1
                        just_finished_are.append(single_result_v_l_score["l_str"])

        logger.info(
            "retro_synthesis: results_v_l_score len(): " + str(len(results_v_l_score)) + " results_v_l_score.pkl at level: " + str(level) + " B"
        )

        logger.info("retro_synthesis: saved_scores: " + str(len(saved_scores)) + " saved_scores.pkl at level: " + str(level) + " B")

        logger.info("retro_synthesis: ------------------------------")

        # Aggregate the branches giving complete RetroSynthesis
        #   score them, at this level this is done for the `just_finished`

        logger.info("retro_synthesis: Aggregate the branches giving complete RetroSynthesis")
        logger.info("retro_synthesis: Aggregate:")
        list_of_lists = [res_["l"] for k, res_ in results_v_l_score_dict.items()]

        # This should not be necessary ... we leave it for the moment as check
        logger.info("retro_synthesis: Aggregate_original_list_of_lists " + str(len(list_of_lists)) + " " + str(list_of_lists))
        list_of_lists = [l1 for l1 in list_of_lists if not list_is_included(l1, list_of_lists)]
        logger.info("retro_synthesis: Aggregate_cleaned_list_of_lists " + str(len(list_of_lists)) + " " + str(list_of_lists))

        for l_str in just_finished_are:
            jfin = results_v_l_score_dict[l_str]["l"][-1]
            res = complete_reactions(list_of_lists, jfin)
            logger.info("retro_synthesis: RESULT jfin/l_str/complete_reactions" + str(jfin) + " " + str(l_str) + " " + str(res))
            length = len(results_v_l_score_dict[l_str]["v"])
            score = results_v_l_score_dict[l_str]["score"]
            score_initial = score * length
            results_v_l_score_dict[l_str]["final_score"] = []
            results_v_l_score_dict[l_str]["complete_reactions"] = []
            # res = [[[],[]]]
            # or res is [
            #             [[1, 2, 3, 6, 7, 10], [1, 2, 3, 6, 9, 11]],
            #             [[1, 2, 3, 6, 7, 10], [1, 2, 3, 6, 9, 12]]
            #           ]
            for r in res:
                # for each alternative the score is initialized
                #   reset to the one of the selected `jfin` -> `l_str`
                score_tmp = score_initial
                # loop on the lists of one of the completed path
                for l1 in r:
                    if len(l1) == 0:
                        continue
                    l_str_local = str(l1)
                    score_tmp = score_tmp + (results_v_l_score_dict[l_str_local]["score"] * len(results_v_l_score_dict[l_str_local]["v"]))
                # save Aggregated results
                results_v_l_score_dict[l_str]["final_score"].append(score_tmp)
                # `r` is a list of lists
                results_v_l_score_dict[l_str]["complete_reactions"].append(r)
                logger.info(
                    "retro_synthesis: aggregated_final_score "
                    + "l_str/final_score/complete_reactions "
                    + str(results_v_l_score_dict[l_str]["final_score"])
                    + " "
                    + str(results_v_l_score_dict[l_str]["v"])
                    + " "
                    + str(results_v_l_score_dict[l_str]["l"])
                    + " "
                    + str(results_v_l_score_dict[l_str]["complete_reactions"])
                )

        logger.info("retro_synthesis: Assemble full smiles reactions n_finished = " + str(n_finished))

        for k_, v_ in results_v_l_score_dict.items():
            # Here the if is not necessary, we passed only once
            #   later it may be useful, but a priori longer path
            #   can finish and open new combinations ...
            # if not 'full_smiles' in results_v_l_score_dict:
            results_v_l_score_dict[k_]["full_smiles"] = get_full_text_sets_of_reactions(v_, results_v_l_score_dict)

            results_v_l_score_dict[k_]["source"] = [hash_str(str(item)) for item in results_v_l_score_dict[k_]["full_smiles"]]

        l_str_present = set([item["l_str"] for item in sources])
        for k_, v_ in results_v_l_score_dict.items():
            for i in range(len(results_v_l_score_dict[k_]["source"])):
                if results_v_l_score_dict[k_]["l_str"] in l_str_present:
                    continue
                dict_ = {}
                dict_["source"] = results_v_l_score_dict[k_]["source"][i]
                dict_["final_score"] = results_v_l_score_dict[k_]["final_score"][i]
                dict_["l_str"] = results_v_l_score_dict[k_]["l_str"]
                sources.append(dict_)

        just_finished_are = []
        logger.info(f"retro_synthesis: End_of_Aggregate at level: {level}")

        # print a little report ...
        logger.info(
            "retro_synthesis: report:"
            + " run_is_finished: "
            + str(run_is_finished)
            + " level: "
            + str(level)
            + " max_levels: "
            + str(max_levels)
            + " n_finished: "
            + str(n_finished)
            + " min_predictions: "
            + str(min_predictions)
        )

        time2 = time.time()
        logger.info("retro_synthesis: TIMING_LOOP_STEP " + str(time2 - time1) + " level: " + str(level) + " n_finished: " + str(n_finished))

        if n_finished > 0 and good_coverage(results_v_l_score_dict, n_opena_at_l1):
            if level < min_levels:
                logger.info(f"retro_synthesis: good_coverage True, but current level is {level} < {min_levels}")
            else:
                logger.info(f"retro_synthesis: good_coverage True, current level is {level}")
                run_is_finished = True

        if level >= max_levels and n_finished >= min_predictions:
            logger.info(
                f"""retro_synthesis: run_is_finished {run_is_finished} level >= max_levels AND n_finished >= min_predictions level: {level} max_levels: {max_levels} n_finished: {n_finished} min_predictions: {min_predictions}"""
            )
            run_is_finished = True
            break

        if n_finished >= max_predictions:
            logger.info(f"""retro_synthesis: run_is_finished {run_is_finished} n_finished >= max_predictions {n_finished} {max_predictions}""")
            run_is_finished = True
            break

        if (time2 - time0) >= 180.0 and n_finished >= min_predictions:
            logger.info(
                f"""retro_synthesis: run_is_finished {run_is_finished} time {time2 - time0} >= 180. seconds AND n_finished >= min_predictions level: {level}, max_levels: {max_levels}, n_finished: {n_finished}, min_predictions: {min_predictions}"""
            )
            run_is_finished = True
            break

        if run_is_finished:
            logger.info(f"retro_synthesis: run_is_finished {run_is_finished} current level {level}")
            break

        if len(leaves_to_be_propagated) == 0:
            logger.info(f"retro_synthesis: len(leaves_to_be_propagated)==0 {len(leaves_to_be_propagated)}")
            break

        if (time2 - time1) > max_step_time:
            run_is_finished = True
            logger.info(f"""retro_synthesis: TIMING_LOOP_max_step_time_END {time2 - time1} max_step_time: {max_step_time}""")
            logger.info(f"retro_synthesis: WALLTIME_run_is_finished: {run_is_finished} level: {level} time2: {time2}")

        if (time2 - time0) > max_total_time:
            run_is_finished = True
            logger.info(f"retro_synthesis: WALLTIME_TIMING_LOOP_max_total_time_END {time2 - time0} max_total_time: {max_total_time}")
            logger.info(f"retro_synthesis: WALLTIME_run_is_finished: {run_is_finished} level: {level} time2: {time2}")

        if run_is_finished:
            break

        if pruning_steps > 1 and (time2 - time0) >= 120.0:
            logger.info(f"retro_synthesis: TIME: {time2-time0} level: {level} reducing pruning_steps: {pruning_steps} to 1")
            pruning_steps = 1

    logger.info(f"retro_synthesis: TIMING_LOOP_STEPS_TOT {(time2 - time0):0.3f} level: {level} n_finished: {n_finished}")

    # END ... SUMMARY

    if run_is_finished or len(leaves_to_be_propagated) == 0:
        logger.info(f"retro_synthesis: END_SUMMARY_run_is_finished {run_is_finished}")
        logger.info(f"retro_synthesis: n_finished = {n_finished}")
        logger.info(f"retro_synthesis: finished level = {level}")
        logger.info(f"retro_synthesis: len(leaves_to_be_propagated): {len(leaves_to_be_propagated)}")
        logger.info("retro_synthesis: get_all_the_finished_reactions ...")
        level_max = get_level_max(tree_dict)
        finished_dict, _ = get_all_the_finished_reactions_up_to(tree_dict, node_reaction_list, level_max)

        # Print all finished reactions (if present)
        logger.info("retro_synthesis: List of all finished reactions")
        #
        # Append - just in case... to make sure order of existing is not changed
        for node in tree_dict:
            if node not in full_node_list_from_tree:
                full_node_list_from_tree.append(node)
        #
        for k, v in finished_dict.items():
            if len(v) > 0:
                for v_ in v:
                    single_result_v_l_score = fetch_score(
                        v_,
                        saved_scores,
                        node_reaction_list,
                        tree_dict,
                        full_node_list_from_tree,
                        map_node2react_smile,
                        True,
                    )
                    score_ = single_result_v_l_score["score"]
                    logger.info(f"retro_synthesis: finished {v_} with score {score_}")

    # Populate `retrosynthesis` is an object of the class Retrosynthesis
    # `saved_confidences` is used to populate the object from `retrosynthesis` Class
    retrosynthesis["routes"] = get_retrosynthesis(results_v_l_score_dict, sources, saved_confidences, saved_classes)
    logger.info("retro_synthesis: retrosynthesis len( _routes_ ): " + str(len(retrosynthesis["routes"])))
    run_is_finished_with_status = "DONE"
    retrosynthesis["status"] = run_is_finished_with_status

    if n_finished == 0:
        # empty // nothing finished
        retrosynthesis["status"] = "ERROR"
        retrosynthesis["routes"] = []

    logger.info("retro_synthesis: retrosynthesis status: " + str(retrosynthesis["status"]))

    time2 = time.time()
    logger.info(f"retro_synthesis: TIME ini/end of retrosynthesis: {time2-time0}")

    return retrosynthesis
