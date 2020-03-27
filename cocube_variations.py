from update_utils import *


def rank_phrase_only(label_phrase_dict, phrase_docid_map, df, labels, i):
    label_entity_dict_list = [label_phrase_dict]
    entity_docid_map_list = [phrase_docid_map]
    filtered_label_entity_dict_list = update_by_percent_together(label_entity_dict_list, entity_docid_map_list, df,
                                                                 labels, i)
    return filtered_label_entity_dict_list[0]


def rank_author_only(label_author_dict, author_docid_map, df, labels, i):
    label_entity_dict_list = [label_author_dict]
    entity_docid_map_list = [author_docid_map]
    filtered_label_entity_dict_list = update_by_percent_together(label_entity_dict_list, entity_docid_map_list, df,
                                                                 labels, i)
    return filtered_label_entity_dict_list[0]


def rank_phrase_author_together(label_phrase_dict, label_author_dict, phrase_docid_map, author_docid_map, df, labels,
                                i, cov="full"):
    label_entity_dict_list = [label_phrase_dict, label_author_dict]
    entity_docid_map_list = [phrase_docid_map, author_docid_map]
    label_phrase_dict, label_author_dict = update_by_percent_together(label_entity_dict_list, entity_docid_map_list,
                                                                      df, labels, i, cov)
    return label_phrase_dict, label_author_dict


def rank_phrase_conf_together(label_phrase_dict, label_conf_dict, phrase_docid_map, venue_docid_map, df, labels,
                              i, cov="full"):
    label_entity_dict_list = [label_phrase_dict, label_conf_dict]
    entity_docid_map_list = [phrase_docid_map, venue_docid_map]
    label_phrase_dict, label_conf_dict = update_by_percent_together(label_entity_dict_list, entity_docid_map_list,
                                                                    df, labels, i, cov)
    return label_phrase_dict, label_conf_dict


def rank_phrase_author_year_together(label_phrase_dict, label_author_dict, label_year_dict, phrase_docid_map,
                                     author_docid_map, year_docid_map, df, labels, i, cov="full"):
    label_entity_dict_list = [label_phrase_dict, label_author_dict, label_year_dict]
    entity_docid_map_list = [phrase_docid_map, author_docid_map, year_docid_map]
    label_phrase_dict, label_author_dict, label_year_dict = update_by_percent_together(label_entity_dict_list,
                                                                                       entity_docid_map_list,
                                                                                       df, labels, i, cov)
    return label_phrase_dict, label_author_dict, label_year_dict


def rank_phrase_author_conf_together(label_phrase_dict, label_author_dict, label_conf_dict, phrase_docid_map,
                                     author_docid_map, venue_docid_map, df, labels, i, cov="full"):
    label_entity_dict_list = [label_phrase_dict, label_author_dict, label_conf_dict]
    entity_docid_map_list = [phrase_docid_map, author_docid_map, venue_docid_map]
    label_phrase_dict, label_author_dict, label_conf_dict = update_by_percent_together(label_entity_dict_list,
                                                                                       entity_docid_map_list,
                                                                                       df, labels, i, cov)
    return label_phrase_dict, label_author_dict, label_conf_dict


def rank_phrase_author_independently(label_phrase_dict, label_author_dict, phrase_docid_map, author_docid_map, df, i):
    label_phrase_dict = update_by_percent(label_phrase_dict, phrase_docid_map, df, i)
    label_author_dict = update_by_percent_with_overlap(label_author_dict, author_docid_map, df, i)
    return label_phrase_dict, label_author_dict


def rank_phrase_author_with_iteration(label_phrase_dict, label_author_dict, df, pred_labels, i):
    label_phrase_dict = update_label_entity_dict_with_iteration(label_phrase_dict, df, pred_labels, i)
    label_author_dict = update_label_entity_dict_with_iteration(label_author_dict, df, pred_labels, i)
    return label_phrase_dict, label_author_dict
