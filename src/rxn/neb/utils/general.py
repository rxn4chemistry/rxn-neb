"""General utilities."""
import ast
from itertools import zip_longest
from typing import Dict, Iterable, List

import regex as re
from loguru import logger


def get_l_from_tree(tree_dict):
    leaves = [node for node in tree_dict if tree_dict[node].is_leaf]
    full_node_list_from_tree = [tree_dict[node].name for node in tree_dict]
    list_of_l = [[full_node_list_from_tree.index(node) for node in get_path_to_node(tree_dict[leaf])] for leaf in leaves]
    return list_of_l


def get_path_to_node(node):
    string_node_list = str(node.path[-1]).replace("Node", "")
    string_node_list = string_node_list.replace("('", "").replace("')", "")
    # NOTE: node separator not ("/"), now it is (",")
    node_list = string_node_list.split(",")
    anchestors = node_list[1:]
    # \\ replace a double backslash with a single backslash
    anchestors = [s.encode().decode("unicode_escape") for s in anchestors]
    return anchestors


def check_alternatives_among(ipos_limit, l_list_single, l_list_of_leaves):
    keep_ = True
    n = 0
    debug_ipos_check_set = set()
    for i in range(len(l_list_of_leaves)):
        check_alternatives_, ipos_check = check_alternatives(l_list_single, l_list_of_leaves[i])
        debug_ipos_check_set.add(ipos_check)
        if check_alternatives_ and ipos_check == ipos_limit:
            keep_ = False  # when False => is not uniq
            n = n + 1
    return keep_, n  # it is unique


def check_alternatives(l_i, l_j):
    m = min(len(l_i), len(l_j))
    for i in range(1, m, 2):
        if not l_i[i] == l_j[i] and l_i[i - 1] == l_j[i - 1] and (i % 2) == 1:
            return True, i
    return False, 0


def select_from_saved_here(x, max_num):
    message_ = "select_from_saved_here: saved_here: " + str(x)
    logger.debug(message_)
    lol = [ast.literal_eval(k) for k in x]
    keep: List[int] = []
    group: Dict[int, Dict[int, List[int]]] = {}
    first_ = False
    for i in range(len(lol)):
        if len(lol[i]) == 1:
            i2 = lol[i][-1]
            first_ = True
            if first_:
                break
        else:
            i2 = lol[i][-2]
        if i2 not in group:
            group[i2] = {}
        group[i2][i] = x[str(lol[i])]

    message_ = "select_from_saved_here: group: " + str(group)
    logger.debug(message_)

    for n_me in range(1, 13):
        if first_:
            break
        keep = []
        for _, xx in group.items():
            best = [k for k, v in sorted(xx.items(), key=lambda item: item[1])][0:n_me]
            for b in best:
                keep.append(b)
        if len(keep) >= max_num:
            break

    nlol = [lol[i] for i in keep]
    if first_:
        nlol = [ast.literal_eval(k) for k, v in sorted(x.items(), key=lambda item: item[1])][0:max_num]
        message_ = "select_from_saved_here_first: n_me/len(keep)/len(x)/nlol:"
        message_ = message_ + " " + str(n_me)
        message_ = message_ + " " + str(len(keep))
        message_ = message_ + " " + str(len(x))
        message_ = message_ + " " + str(nlol)
        logger.info(message_)

    return nlol


def select_prefix_to_keep(node_reaction_list, save_here):
    try:
        s_ = [k for k, v in sorted(save_here.items(), key=lambda item: item[1])][0]
        list_best = ast.literal_eval(s_)
        id_ = list_best[0]
        ns_ = node_reaction_list[id_]
        prefix_ = re.sub(r"(R\d+)___.*", r"\1", ns_)
        logger.info(f"select_prefix_to_keep: keep_this_prefix: {prefix_}")
    except Exception:
        prefix_ = ""
        logger.info("select_prefix_to_keep: keep_this_prefix: FAILS")
        logger.debug("select_prefix_to_keep: keep_this_prefix: FAILS")
    return prefix_


def batcher(iterable: Iterable, batch_size: int = 32):
    """Generate batches from an iterable.

    Args:
        iterable: iterable to batch.
        batch_size: batch_size.

    Returns:
        batches of the desired size.
    """
    batched_iterable = [iter(iterable)] * batch_size
    for batch in zip_longest(*batched_iterable, fillvalue=None):
        yield [element for element in batch if element is not None]
