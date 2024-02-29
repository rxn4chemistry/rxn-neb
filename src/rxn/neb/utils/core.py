"""Core utilities."""
import hashlib
import itertools
from copy import deepcopy
from typing import Any, Dict, List, Set

import numpy as np
import regex as re
from anytree import Node as NodeBase
from loguru import logger

from ..availability import is_available
from ..fingerprints import compute_rxnfp as rxnfp
from ..single_step_predictions import run_rxn_prediction
from .general import check_alternatives, get_l_from_tree, get_path_to_node
from .smiles import from_complex_smile_to_fragments, try_further


class Node(NodeBase):
    separator = ","


def one_smile2fp(smile, pca, fingerprints_model):
    fp = rxnfp(smile, fingerprints_model=fingerprints_model)
    array = np.array([fp], dtype="float32")
    array = pca.transform(array)
    array = array.reshape(-1)
    fp = array.tolist()
    # print('one_smile2fp len(fp):', len(fp))
    return fp


def smiles2fp(smiles, pca, fingerprints_model):
    fps = [rxnfp(smile, fingerprints_model=fingerprints_model) for smile in smiles]
    array = np.array(fps, dtype="float32")
    # PCA reduction
    array = pca.transform(array)
    return array


def check_reaction(keep, r, reaction_smiles_txt, reaction_smiles_fp):
    #
    # Check reactions runs over a small number of reactions,
    #  we keep the distances in the original space. It is
    #  also more accurate.
    #
    logic = True
    i = reaction_smiles_txt.index(r)
    a = reaction_smiles_fp[i]
    for r_ in keep:
        j = reaction_smiles_txt.index(r_)
        b = reaction_smiles_fp[j]
        d = np.sum((a - b) ** 2, axis=0)
        if d < 2.0:
            return False
    return logic


def update_reaction_smiles_from_tree(tree_dict, node2rs, node_list):
    my_depths = set([tree_dict[node].depth for node in tree_dict])
    my_depths_max = max(my_depths) + 1
    my_select = [i for i in range(1, my_depths_max, 2)]

    for node in tree_dict:
        if tree_dict[node].depth in my_select:
            tmp = get_path_to_node(tree_dict[node].ancestors[-1])
            if len(tmp) >= 2:
                # ancestor is a product (or an interm. product, not a reaction)
                # tmp[-1] should point to an even element of the list `tmp`
                ancestor = re.sub(r"^.*?___", "", tmp[-1])
            else:
                ancestor = re.sub(r"^.*?___", "", tmp[0])
            precursors = re.sub(r"^.*?___", "", node)
            if "NONE" in precursors:
                continue  # here OK either continue or pass
            else:
                s_ = precursors + ">>" + ancestor
                if node not in node2rs:
                    node2rs[node] = s_
                    node_list.append(node)
    return node2rs, node_list


def get_compounds_in_list(list_):
    comp_list = []
    for name_ in list_:
        if "_MORE___" in name_ or "ROOT___" in name_ or "_END___" in name_:
            smile = re.sub(r"^.*?___", "", name_)
            # pref_ = re.sub(r"___.*", "", name_)
            comp_list.append(smile)
    return comp_list


def add_to_saved_reactions(prod, retro_predictions: List[str], saved_reactions: Dict[str, Set[str]]) -> Dict[str, Set[str]]:
    if prod not in saved_reactions:
        saved_reactions[prod] = set()
    for retro in retro_predictions:
        saved_reactions[prod].add(retro)
    return saved_reactions


def initialize_tree(product, retro_predictions_good):
    tree_dict = {}
    root_ = "ROOT" + "___" + str(product)
    tree_dict[root_] = Node(root_)
    for i in range(len(retro_predictions_good)):
        child = retro_predictions_good[i]
        name_ = "R" + str(i) + "___" + str(child)
        tree_dict[name_] = Node(name_, parent=tree_dict[root_])
        smile_split, s_ = from_complex_smile_to_fragments(child)
        j = 0
        for s in smile_split:
            if is_available(s) or try_further(s):
                name_c = "R" + str(i) + "_C" + str(j) + "__END" + "___" + str(s)
                j = j + 1
            else:
                name_c = "R" + str(i) + "_C" + str(j) + "_MORE" + "___" + str(s)
                j = j + 1
            tree_dict[name_c] = Node(name_c, parent=tree_dict[name_])
    return tree_dict


def propagate_leaf(
    leaf,
    tree_dict,
    saved_reactions,
    clean,
    pca_model,
    saved_confidences,
    saved_classes,
    fw_model,
    re_model,
    fingerprints_model,
):
    product = re.sub(r"\w+___", "", leaf)

    if product in saved_reactions:
        retro_predictions_good = [item for item in saved_reactions[product]]
    else:
        retro_predictions_good = run_rxn_prediction(product, fw_model=fw_model, re_model=re_model)

    saved_reactions = add_to_saved_reactions(product, retro_predictions_good, saved_reactions)

    if clean and len(retro_predictions_good) > 0:
        # Sort the list of strings by string len
        sorted_retro_predictions_good = sorted(retro_predictions_good, key=lambda el: len(el))

        # Filter the duplicates
        r_smiles = [r + ">>" + product for r in sorted_retro_predictions_good]
        fp = smiles2fp(r_smiles, pca_model, fingerprints_model=fingerprints_model)
        keep = []
        keep.append(r_smiles[0])
        n = len(r_smiles)
        for i in range(1, n):
            r = r_smiles[i]
            if check_reaction(keep, r, r_smiles, fp):
                keep.append(r)

        retro_predictions_good = [r.split(">>")[0] for r in keep]

    if len(retro_predictions_good) == 0:
        child = "NONE"
        leaf_n = re.sub(r"_MORE.*", "", leaf)
        name_ = leaf_n + "_R0___NONE"
        tree_dict[name_] = Node(name_, parent=tree_dict[leaf])
    else:
        for j in range(len(retro_predictions_good)):
            child = retro_predictions_good[j]
            leaf_n = re.sub(r"_MORE.*", "", leaf)
            name_ = leaf_n + "_R" + str(j) + "___" + str(child)
            if name_ in tree_dict:
                logger.debug("propagate_leaf: skip")
            else:
                tree_dict[name_] = Node(name_, parent=tree_dict[leaf])
    return tree_dict, saved_reactions, saved_confidences, saved_classes


def split_precursors_with_wait(tree_dict, my_list, node_wait, my_exclude):
    for ime in range(len(my_list)):
        name_ = my_list[ime]
        if "__NONE" in name_:
            continue
        node = tree_dict[name_]
        list_ = get_path_to_node(node)
        ancestors_compounds = get_compounds_in_list(list_)
        smile = re.sub(r"^.*?___", "", name_)
        pref_ = re.sub(r"___.*", "", name_)
        smile_split, s_ = from_complex_smile_to_fragments(smile)
        j = 0
        # This loop is to check if there is one repeated fragment
        #    close loops (if one is stop, all are tagged as stop)
        stop_ = False
        for s in smile_split:
            if s in ancestors_compounds:
                if s in my_exclude:
                    stop_ = True
                elif is_available(s) or try_further(s):
                    pass
                elif len(s) <= 7:  # [Na+]
                    pass
                else:
                    stop_ = True
        for s in smile_split:
            if is_available(s) or try_further(s):
                name_c = pref_ + "_C" + str(j) + "__END" + "___" + str(s)
                j = j + 1
            else:
                name_c = pref_ + "_C" + str(j) + "_MORE" + "___" + str(s)
                j = j + 1
            if name_c in tree_dict:
                raise Exception("split_precursors_with_wait: error: dist_pair_lists_max_far len(list_a) != len(list_b)")
            elif stop_:
                name_c = name_c.replace("__END", "_STOP")
                name_c = name_c.replace("_MORE", "_STOP")
                tree_dict[name_c] = Node(name_c, parent=tree_dict[name_])
            else:
                tree_dict[name_c] = Node(name_c, parent=tree_dict[name_])

    # node to wait
    for name_ in node_wait:
        if "__NONE" in name_:
            continue
        node = tree_dict[name_]
        list_ = get_path_to_node(node)
        ancestors_compounds = get_compounds_in_list(list_)
        smile = re.sub(r"^.*?___", "", name_)
        pref_ = re.sub(r"___.*", "", name_)
        smile_split, s_ = from_complex_smile_to_fragments(smile)
        j = 0
        # This loop is to check if there is one repeated fragment
        #    close loops (if one is stop, all are tagged as stop)
        stop_ = False
        for s in smile_split:
            if s in ancestors_compounds:
                if s in my_exclude:
                    stop_ = True
                elif is_available(s) or try_further(s):
                    pass
                else:
                    stop_ = True

        for s in smile_split:
            if is_available(s) or try_further(s):
                name_c = pref_ + "_C" + str(j) + "__END" + "___" + str(s)
                j = j + 1
            else:
                name_c = pref_ + "_C" + str(j) + "_WAIT" + "___" + str(s)
                j = j + 1
            if name_c in tree_dict:
                pass
            elif stop_:
                name_c = name_c.replace("__END", "_STOP")
                name_c = name_c.replace("_MORE", "_STOP")
                tree_dict[name_c] = Node(name_c, parent=tree_dict[name_])
            else:
                tree_dict[name_c] = Node(name_c, parent=tree_dict[name_])
    return tree_dict


def compute_reaction_smiles_txt2fp_dict(reaction_smiles_txt2fp_dict, reaction_smiles_txt, pca, fingerprints_model):
    for s_ in reaction_smiles_txt:
        if s_ not in reaction_smiles_txt2fp_dict:
            reaction_smiles_txt2fp_dict[s_] = one_smile2fp(s_, pca, fingerprints_model)
    return reaction_smiles_txt2fp_dict


def check_reaction_node_id(keep, r, map_node2react_smile, reaction_smiles_txt2fp_dict):
    logic = True
    a = np.array(reaction_smiles_txt2fp_dict[map_node2react_smile[r]])
    for r_ in keep:
        b = np.array(reaction_smiles_txt2fp_dict[map_node2react_smile[r_]])
        d = np.sum((a - b) ** 2, axis=0)
        if d < 2.0:
            return False
    return logic


def filter_similar_node_id(list_of_list, node_reaction_list, map_node2react_smile, reaction_smiles_txt2fp_dict):
    list_uniq = []
    first: Dict[int, List[str]] = {}
    for l1 in list_of_list:
        r = node_reaction_list[l1[-1]]
        if l1[0] not in first:
            first[l1[0]] = []
            first[l1[0]].append(r)
            list_uniq.append(l1)
            continue
        if check_reaction_node_id(first[l1[0]], r, map_node2react_smile, reaction_smiles_txt2fp_dict):
            first[l1[0]].append(r)
            list_uniq.append(l1)
        else:
            pass
    logger.debug("filter_similar_node_id: len(list_uniq) " + str(len(list_uniq)))
    return list_uniq


def filter_similar_node_id_with_tree(
    tree_dict,
    list_of_list,
    node_reaction_list,
    map_node2react_smile,
    reaction_smiles_txt2fp_dict,
):
    full_node_list_from_tree_local = [tree_dict[node].name for node in tree_dict]
    list_uniq = []
    first: Dict[int, List[str]] = {}
    for l1 in list_of_list:
        l2 = [full_node_list_from_tree_local.index(i) for i in get_path_to_node(tree_dict[node_reaction_list[l1[-1]]])]
        if len(l2) > 1:
            l0 = l2[-2]
        else:
            l0 = 0
        r = node_reaction_list[l1[-1]]
        if l0 not in first:
            first[l0] = []
            first[l0].append(r)
            list_uniq.append(l1)
            continue
        if check_reaction_node_id(first[l0], r, map_node2react_smile, reaction_smiles_txt2fp_dict):
            first[l0].append(r)
            list_uniq.append(l1)
        else:
            pass
    return list_uniq


def select_l_nb_node_id(tree_dict, node_reaction_list, l_select):
    # Compute the list of id of the reactions at given level
    # l_select=3
    if l_select not in [2 * i + 1 for i in range(100)]:
        return []
    list_of_list_of_2_inds = []
    for node in tree_dict:
        if "NONE" in node:
            continue
        if tree_dict[node].depth == l_select:
            tmp = get_path_to_node(tree_dict[node])
            l_ = [node_reaction_list.index(n_) for n_ in tmp if n_ in node_reaction_list]
            # correction to avoid empty lists appended ... [[],[],[],...]
            if len(l_) > 0:
                list_of_list_of_2_inds.append(l_)
    return list_of_list_of_2_inds


def select_l_nb_filter_node_id(tree_dict, node_reaction_list, l_select, list_filter):
    # Compute the list of id of the reactions at given level
    if l_select not in [2 * i + 1 for i in range(100)]:
        return []
    if len(list_filter) > 0:
        nlen = len(list_filter[0])
    else:
        nlen = -1
    list_of_list = []
    for node in tree_dict:
        if "NONE" in node:
            continue
        if tree_dict[node].depth == l_select:
            tmp = get_path_to_node(tree_dict[node])
            list_of_inds = [node_reaction_list.index(n_) for n_ in tmp if n_ in node_reaction_list]
            if nlen > 0:
                if list_of_inds[0:nlen] in list_filter:
                    # correction to avoid empty lists appended ... [[],[],[],...]
                    if len(list_of_inds) > 0:
                        list_of_list.append(list_of_inds)
                else:
                    pass
            else:
                # correction to avoid empty lists appended ... [[],[],[],...]
                if len(list_of_inds) > 0:
                    list_of_list.append(list_of_inds)
    return list_of_list


def get_score_for_a_single_path(
    list_of_steps,
    tree_data,
    node_reaction_list,
    map_node2react_smile,
    reaction_smiles_txt2fp_dict,
    saved_scores,
    max_dist_along_cutoff,
    n_neighbors: int = 5,
    max_length_path_neb: int = 15,
):
    lmax = max_length_path_neb
    nmax = len(list_of_steps)

    # Important: `list_str`: this is changed in the loop
    #   only at the end of the loop is set equal to the full list
    #   to be saved in the 'global' variable.

    nl_count = 0
    sum_min_dist = 0.0
    for ib1 in range(nmax):
        for j in range(1, lmax):
            ib2 = ib1 + j
            # this `ib2 > nmax` should be without equal to include the last step
            if ib2 > nmax:
                break
            list_ = [list_of_steps[i] for i in range(ib1, ib2)]
            list_str = str(list_)  # this is to make an id of `list_`

            if list_str in saved_scores["part"]:
                min_dist = saved_scores["part"][list_str]
            else:
                # `my_fps_series` is a list of fps ...
                my_fps_series = [reaction_smiles_txt2fp_dict[map_node2react_smile[node_reaction_list[i]]] for i in list_]
                my_data = np.array(my_fps_series)
                my_data = my_data.reshape(1, j * 16)
                dist, ind = tree_data[j].query(my_data, k=n_neighbors)
                #
                # DISTANCE KDTree
                # min_dist = np.mean(dist)
                min_dist = np.mean(np.mean(dist * dist, axis=1))
                saved_scores["orig_part_dists"][list_str] = [dist[0][i] * dist[0][i] for i in range(dist.shape[1])]
                saved_scores["orig_part"][list_str] = min_dist
                saved_scores["orig_part_inds"][list_str] = ind
                # CUT_OFF_min_dist, below min_dist not further used,
                # not check one by one vs. max_dist_along_cutoff
                if min_dist > max_dist_along_cutoff * len(list_):
                    min_dist = max_dist_along_cutoff * len(list_)
                saved_scores["part"][list_str] = min_dist
                saved_scores["part_inds"][list_str] = ind
                saved_scores["part_len_steps"][list_str] = len(list_)
                saved_scores["part_steps"][list_str] = list_

            saved_scores["local"][list_str] = min_dist / len(list_)
            nl_count = nl_count + 1
            sum_min_dist = sum_min_dist + min_dist / len(list_)

    # Important reset list_str=str(list_of_steps)
    # Now `list_str` is the string of the full `list_of_steps`
    list_str = str(list_of_steps)
    if nl_count > 0:
        sum_min_dist = sum_min_dist / nl_count
        saved_scores["global"][list_str] = sum_min_dist
    else:
        pass
    return saved_scores


def get_score_for_many_single_path(
    list_of_list_of_steps,
    tree_data,
    node_reaction_list,
    map_node2react_smile,
    reaction_smiles_txt2fp_dict,
    saved_scores,
    max_dist_along_cutoff,
):
    saved_here = {}
    for i_ in range(len(list_of_list_of_steps)):
        list_of_steps = list_of_list_of_steps[i_]
        # print('get_score_for_many_single_path', i_, list_of_steps)
        # list_of_steps = [it for it in list_of_list_of_steps[i_]]
        if len(list_of_steps) < 1:
            continue
        saved_scores = get_score_for_a_single_path(
            list_of_steps,
            tree_data,
            node_reaction_list,
            map_node2react_smile,
            reaction_smiles_txt2fp_dict,
            saved_scores,
            max_dist_along_cutoff,
        )
        saved_here[str(list_of_steps)] = saved_scores["global"][str(list_of_steps)]
    return saved_scores, saved_here


def get_level_max(tree_dict):
    depth_set = set([tree_dict[node].depth for node in tree_dict])
    level_max = max(depth_set)
    return level_max


def remove_nodes(new_tree_dict):
    # list of nodes to be removed with or without parent
    node_del = []
    node_del_with_parent = []
    for node in new_tree_dict:
        if new_tree_dict[node].depth == 0:
            if "ROOT" not in node:
                continue
        number = len([c for c in [i_.name for i_ in new_tree_dict[node].children] if c in new_tree_dict])
        if number > 0:
            continue
        if "STOP" in node or "MORE" in node or "WAIT" in node:
            node_del_with_parent.append(node)
        elif "END" in node:
            pass  # this is good
        else:
            # this is a reaction not continued
            node_del.append(node)

    # nullify `node_del`
    for node in node_del:
        new_tree_dict[node] = Node(node, parent=None)
    for node in node_del:
        del new_tree_dict[node]

    # nullify `node_del_with_parent`
    more_to_del = []
    for node in node_del_with_parent:
        n = new_tree_dict[node].parent
        new_tree_dict[node] = Node(node, parent=None)
        new_tree_dict[n.name] = Node(n.name, parent=None)
        more_to_del.append(n.name)
    more_and_more_to_del = [re.sub(r"(^.*?)___.*$", r"\1", n_) for n_ in more_to_del]

    for node in node_del_with_parent:
        del new_tree_dict[node]

    for node in more_to_del:
        if node in new_tree_dict:
            del new_tree_dict[node]

    node_del = []
    for prefix_without_underscore in more_and_more_to_del:
        prefix_ = prefix_without_underscore + "_"
        for node in new_tree_dict:
            if "ROOT" in node:
                continue
            if node[: len(prefix_)] == prefix_:
                node_del.append(node)

    # nullify `node_del`
    for node in node_del:
        if node in new_tree_dict:
            new_tree_dict[node] = Node(node, parent=None)
    for node in node_del:
        if node in new_tree_dict:
            del new_tree_dict[node]

    return new_tree_dict


def get_all_the_finished_reactions_up_to(tree_dict, node_reaction_list, level_max):
    if level_max in [2 * i + 1 for i in range(100)]:
        level_max = level_max + 1
    logger.debug(f"get_all_the_finished_reactions_up_to: level_max: {level_max}")

    # Make a copy of the tree_dict
    new_tree_dict = deepcopy(tree_dict)

    # Remove nodes above level_max (L>=level_max+1)
    delete = []
    for node in new_tree_dict:
        if new_tree_dict[node].depth >= level_max + 1:
            delete.append(node)

    for node in delete:
        new_tree_dict[node] = Node(node, parent=None)
    # May consider to delete the key
    for node in delete:
        del new_tree_dict[node]

    nf = -1
    ni = len(new_tree_dict)
    while not ni == nf:
        ni = len(new_tree_dict)
        new_tree_dict = remove_nodes(new_tree_dict)
        nf = len(new_tree_dict)

    # The exclude list is to remove the shortest pathways from the long ones
    dict_of_list_of_x_inds: Dict[int, List[int]] = {}
    if level_max in [2 * i + 1 for i in range(100)]:
        my_l = level_max
    else:
        my_l = level_max - 1

    for i_level in range(1, my_l + 1, 2):
        dict_of_list_of_x_inds[i_level] = []
        list_of_list_of_x_inds_ = select_l_nb_node_id(new_tree_dict, node_reaction_list, i_level)
        # We need to check for the last are good
        for l1 in list_of_list_of_x_inds_:
            bad = False
            n = l1[-1]
            for item in new_tree_dict[node_reaction_list[n]].children:
                if "MORE" in item.name or "STOP" in item.name:
                    bad = True
                    break
            if not bad:
                dict_of_list_of_x_inds[i_level].append(l1)

    return dict_of_list_of_x_inds, new_tree_dict


def check_sanity_tree(tree_dict, leaves_purged_set, verbose: bool = True):
    sanity_status: Dict[str, Any] = {
        "check": False,
        "list_of_l_more_to_stop": [],
        "list_of_node_more_to_stop": [],
        "more_status": {},
        "to_complete": {},
    }

    leaves = [node for node in tree_dict if (tree_dict[node].is_leaf and node not in leaves_purged_set)]
    full_node_list_from_tree = [tree_dict[node].name for node in tree_dict]
    list_of_l = get_l_from_tree(tree_dict)

    more_ = set([node for node in leaves if "MORE" in node])
    end_ = set([node for node in leaves if "END" in node])
    wait_ = set([node for node in leaves if "WAIT" in node])
    stop_ = set([node for node in leaves if "STOP" in node])
    none_ = set([node for node in leaves if "NONE" in node])

    stop_ = stop_.union(leaves_purged_set)

    others_ = set()
    for node_ in leaves:
        if node_ in more_:
            continue
        if node_ in end_:
            continue
        if node_ in wait_:
            continue
        if node_ in stop_:
            continue
        if node_ in none_:
            continue
        others_.add(node_)

    bad_node = none_.union(stop_)
    if verbose:
        logger.info("check_sanity_tree info")
        logger.info(f"check_sanity_tree: {len(others_)} len_others_ {others_}")
        logger.info(f"check_sanity_tree: {len(more_)} more_")
        logger.info(f"check_sanity_tree: {len(end_)} end_")
        logger.info(f"check_sanity_tree: {len(wait_)} wait_")
        logger.info(f"check_sanity_tree: {len(stop_)} stop_")
        logger.info(f"check_sanity_tree: {len(none_)} none_")
        logger.info(f"check_sanity_tree: {len(bad_node)} bad_node")

    my_more = [node for node in full_node_list_from_tree if node in more_]
    for node_more in my_more:
        sanity_status["to_complete"][node_more] = set()
        ime = full_node_list_from_tree.index(node_more)
        r = complete_reactions(list_of_l, ime)
        status_ = []
        check_of_what_could_be = set()
        for i in range(len(r)):
            ok = True
            a_ = [ll[-1] for ll in r[i] if len(ll) > 0]
            for e in a_:
                check_of_what_could_be.add(e)
                if full_node_list_from_tree[e] in bad_node:
                    ok = False
                    break
            status_.append(ok)
            if ok:
                for e in a_:
                    el = full_node_list_from_tree[e]
                    if el in more_ or el in wait_:
                        sanity_status["to_complete"][node_more].add(el)
        if any(status_):
            logger.info(f"check_sanity_tree: {ime} STATUS GOOD {node_more}")
            sanity_status["more_status"][node_more] = "GOOD"
        else:
            logger.info(f"check_sanity_tree: {ime} STATUS BAD {node_more}")
            sanity_status["more_status"][node_more] = "BAD"
            if ime not in sanity_status["list_of_l_more_to_stop"]:
                sanity_status["list_of_l_more_to_stop"].append(ime)
            if node_more not in sanity_status["list_of_node_more_to_stop"]:
                sanity_status["list_of_node_more_to_stop"].append(node_more)

    sanity_status["check"] = True

    return sanity_status


def fetch_score(
    r_,
    saved_scores,
    node_reaction_list,
    tree_dict,
    full_node_list_from_tree,
    map_node2react_smile,
    verbose,
):
    r_str = str(r_)
    if r_str in saved_scores["global"]:
        score_ = saved_scores["global"][r_str]
    else:
        score_ = "N/A"
    n = r_[-1]
    res = get_path_to_node(tree_dict[node_reaction_list[n]])
    l1 = [full_node_list_from_tree.index(i) for i in res]
    message_ = "fetch_score: " + str(len(r_)) + " ==> " + str(r_)
    message_ = message_ + " >> " + str(l1) + " score (this one): " + str(score_)
    logger.debug(message_)
    if verbose:
        text = []
        for ik in range(len(r_)):
            nik = node_reaction_list[r_[ik]]
            t = map_node2react_smile[nik]
            text.append(t)

    single_result_v_l_score = {}
    single_result_v_l_score["score"] = score_
    single_result_v_l_score["v"] = [i for i in r_]
    single_result_v_l_score["l"] = [i for i in l1]
    single_result_v_l_score["v_str"] = str(r_)
    single_result_v_l_score["l_str"] = str(l1)
    single_result_v_l_score["text"] = [t for t in text]

    return single_result_v_l_score


def list_is_included(my_list, list_of_lists):
    s_my_list = "-".join([str(i) for i in my_list]) + "-"
    s_list_of_lists = set(["-".join([str(i) for i in l1]) + "-" for l1 in list_of_lists if not my_list == l1])
    # print('s_list_of_lists', s_list_of_lists)
    for s in s_list_of_lists:
        # print('s_my_list vs. s', s_my_list, s)
        if s_my_list in s:
            return True
    return False


def complete_reactions(list_of_lists, jfin):
    #
    # list_of_lists=[ [1,2], [1,2,3,6], [1,2,5,29],
    #                 [1,2,3,6,7,10], [1,2,3,6,9,11], [1,2,3,6,9,12] ]
    #
    # list_of_lists should be updated at the given level (level of jfin)
    # jfin an id of the selected finished reactions we want to complete
    # jfin is the last id in one of the lists in `list_of_lists`
    #
    a = []
    necessary_to_completion: List[List[int]] = []
    for i in range(len(list_of_lists)):
        my_list = list_of_lists[i]
        if jfin == my_list[-1]:
            logger.debug(f"complete_reactions: {jfin}, my_list: {my_list}")
            if len(my_list) < 4:
                # print("strange, because here we are at least in a two-steps reaction")
                continue

            # print('check:', list_is_included(my_list, list_of_lists))
            if list_is_included(my_list, list_of_lists):
                logger.debug(f"complete_reactions: error in complete_reactions {my_list} list_is_included")
                logger.error("complete_reactions: error in complete_reactions list_is_included")
                raise Exception("Error in complete_reactions list_is_included")

            necessary_to_completion = []
            for isplit in range(1, len(my_list), 2):
                # parent_m2 = my_list[isplit]
                # print('parent_m2:', parent_m2, 'with isplit:', isplit)
                for k in range(len(list_of_lists)):
                    k_list = list_of_lists[k]
                    if len(k_list) <= isplit + 1:
                        continue
                    if my_list == k_list:
                        continue
                    # if not len(my_list)==len(k_list): continue
                    # if my_list[-2]==k_list[-2]: continue
                    if my_list[isplit] == k_list[isplit]:
                        if my_list[isplit + 1] != k_list[isplit + 1]:
                            necessary_to_completion.append(k_list)
                            a.append(k_list[-2])
                            # print('necessary to completion:', my_list, k_list)
    # print("complete_reactions_necessary_to_completion:", necessary_to_completion)
    # Check which routes are alternatives and which are complemetary
    list_of_lists = [l1 for l1 in necessary_to_completion]
    # making groups
    ngr = 0
    groups = {}
    nl = len(list_of_lists)
    groups[ngr] = [
        0,
    ]
    for i in range(1, nl):
        alt_ = False
        for k, v in groups.items():
            for vv in v:
                check_alternatives_, _ = check_alternatives(list_of_lists[vv], list_of_lists[i])
                if check_alternatives_:
                    groups[k].append(i)
                    alt_ = True
                    break
            if alt_:
                break
        if not alt_:
            ngr = ngr + 1
            groups[ngr] = [
                i,
            ]
    # ---
    logger.debug("complete_reactions: complete_all_reactions_groups " + str(groups))
    l1 = [v for k, v in groups.items()]
    # print("complete_all_reactions_groups:", l1)
    # ---
    combinations_for_completion = [list(i) for i in list(itertools.product(*l1))]
    # print(
    #     "complete_reactions_combinations_for_completion:", combinations_for_completion
    # )
    if len(necessary_to_completion) > 0:
        results = [[necessary_to_completion[ll] for ll in l1] for l1 in combinations_for_completion]
    else:
        results = [[[]]]
    return results


def good_coverage(results_v_l_score_dict, n_initial):
    a = [v["v"][0] for k, v in results_v_l_score_dict.items()]
    b = [v["v"][0] for k, v in results_v_l_score_dict.items() if min(v["final_score"]) < 40.0]
    if len(set(b)) >= int(0.50 * n_initial):
        return True
    if len(set(b)) >= int(0.33 * n_initial) and len(b) >= n_initial:
        return True
    if len(set(a)) >= int(0.66 * n_initial):
        return True
    if len(set(a)) >= int(0.50 * n_initial) and len(a) >= 2 * n_initial:
        return True
    return False


def get_full_text_sets_of_reactions(v, res):
    # For each value of the result dictionary `res`
    #   make the complete list of smiles associated
    #   to that reaction
    #   `texts` is at the end a list of sorted lists
    texts = []
    n_reactions = len(v["final_score"])
    n_check = len(v["complete_reactions"])
    if not n_reactions == n_check:
        logger.debug("get_full_text_sets_of_reactions: error: why len(final_score) is not equal to len(complete_reactions)?")
        logger.error("get_full_text_sets_of_reactions: Error: len(final_score) != len(complete_reactions)")
        raise Exception("Error: len(final_score) != len(complete_reactions)")
    for n in range(n_reactions):
        full_text = set(v["text"])
        full_text_sorted_list = sorted(list(full_text))
        if not str(v["complete_reactions"]) == "[[[]]]":
            k_tot = len(v["complete_reactions"][n])
            for k in range(k_tot):
                l_str = str(v["complete_reactions"][n][k])
                for s in res[l_str]["text"]:
                    full_text.add(s)
            full_text_sorted_list = sorted(list(full_text))
        # Note: even in case of "[[[]]]" there should be the following append!
        texts.append(full_text_sorted_list)
    # print('debug_get_full_text_sets_of_reactions: v, len(texts), texts',
    #       v, len(texts), texts)
    if len(texts) == 0:
        logger.debug("get_full_text_sets_of_reactions: error: why len(texts) is zero?")
        logger.error("get_full_text_sets_of_reactions: error: len(texts) is zero?")
        raise Exception("get_full_text_sets_of_reactions: error: len(texts) is zero?")
    return texts


def hash_str(t):
    return hashlib.sha1(str(t).encode("utf-8")).hexdigest()


def reshape_score(x):
    return 1.0 / (1.0 + np.log10(1.0 + x))


def get_retrosynthesis(
    results_v_l_score_dict,
    sources,
    saved_confidences,
    saved_classes,
):
    logger.debug(f"get_retrosynthesis: len(saved_confidences): {len(saved_confidences)}")
    logger.debug(f"get_retrosynthesis: saved_confidences: {saved_confidences}")
    d1 = {}
    for k_, v_ in results_v_l_score_dict.items():
        for i in range(len(v_["source"])):
            source_ = v_["source"][i]
            smiles_ = sorted(list(v_["full_smiles"][i]))
            d1[source_] = smiles_
    d2 = {k["source"]: k["final_score"] for k in sources}
    # Limit to the top N scores (here lower is better)
    n_max = 100
    if len(d2) < n_max:
        n_max = len(d2)
    if n_max <= 10:
        topn_threshold = 9999999.0
        logger.debug(f"get_retrosynthesis: topn_threshold N={n_max}: {topn_threshold}")
    else:
        topn_threshold = sorted([v for k, v in d2.items()])[n_max - 1]
        logger.debug(f"get_retrosynthesis: topn_threshold N={n_max}: {topn_threshold}")
    #
    retrosynthesis = []
    for source_ in d1:
        if source_ not in d2:
            logger.debug(f"get_retrosynthesis: Strange source_ not in d2: {source_}")
            continue
        if n_max > 10 and d2[source_] > topn_threshold:
            continue  # This is just to speed up and keep topN N=30 above
        logger.debug(f"get_retrosynthesis: summary_single_retro_result: source: {source_}")
        #
        single = {}
        #
        # add `source` to single to tag it
        single["source"] = source_
        single["reactions"] = []
        logger.debug("get_retrosynthesis: populating the reactions")
        for reaction_smiles in d1[source_]:
            d_ = {}
            d_["reaction_smiles"] = reaction_smiles
            logger.debug(f"get_retrosynthesis: reaction_smiles: {reaction_smiles}")
            if reaction_smiles in saved_confidences:
                d_["confidence"] = saved_confidences[reaction_smiles]
                logger.debug("get_retrosynthesis: confidence: " + str(d_["confidence"]))
            if reaction_smiles in saved_classes:
                d_["reaction_class"] = saved_classes[reaction_smiles]
            single["reactions"].append(d_)
        # -----
        if len(d1[source_]) < 1:
            logger.debug("get_retrosynthesis: Strange no reactions for " + str(source_) + " " + str(single["reactions"]))
        # here score in changed score = 1/(1+log10(1+score))
        single["optimization_score"] = reshape_score(d2[source_])
        retrosynthesis.append(single)

    # sort by score: after applying reshape_score, high score is better than low score
    # reshape_score: see above changes score x -> 1/(1+log10(1+x))
    logger.debug(f"get_retrosynthesis: len(retrosynthesis) before applying cut {len(retrosynthesis)}")
    # limit max number of returned results
    retrosynthesis = sorted(retrosynthesis, key=lambda d: d["optimization_score"], reverse=True)[0 : min(len(retrosynthesis), 30)]
    logger.debug(f"get_retrosynthesis: len(retrosynthesis) after applying cut {len(retrosynthesis)}")
    return retrosynthesis
