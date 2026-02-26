def compare_quick_null(tree, snv_id, snv_dict, 
                       cna_tree, snv_dataset, snv_idx,
                       alpha, p0, p1_fp_mode):
    from ..likelihood import quick_null_stats as old_qn
    from ..likelihood_new.quick_null_stats_new import quick_null_stats_new
    
    old = old_qn(tree, snv_id, snv_dict, p0, alpha, p1_fp_mode)
    new = quick_null_stats_new(cna_tree, snv_dataset, snv_idx, alpha, p0, p1_fp_mode)

    print("old:", old)
    print("new:", new)

    return old, new