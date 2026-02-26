def test_compare_batch(tree, snv_dict, snv_ids,
                       cna_tree, snv_dataset, transitions,
                       batch_size=20):
    import numpy as np
    from ..likelihood import locus_loglik_batch as old_batch
    from ..likelihood_new.locus_loglik_batch_new import locus_loglik_batch_new

    # ensure Python list
    if not isinstance(snv_ids, (list, tuple)):
        snv_ids = list(snv_ids)

    some_snvs = snv_ids[:batch_size]

    # ---- OLD ----
    old_out = old_batch(tree, some_snvs, snv_dict)

    # ---- NEW ----
    snv_idx_map = {snv_id: i for i, snv_id in enumerate(snv_dataset.snv_ids)}
    some_indices = [snv_idx_map[s] for s in some_snvs]

    new_out = locus_loglik_batch_new(cna_tree, snv_dataset, transitions, some_indices)

    # ---- COMPARE ----
    for snv in some_snvs:
        print(f"\nComparing SNV {snv}")
        old_lik = old_out[snv]
        new_lik = new_out[snv]

        for node_obj in old_lik.keys():

            o = old_lik[node_obj]

            # convert ETE node → integer index
            if node_obj is None:
                node_idx = None
            else:
                node_idx = cna_tree.ete_to_idx[node_obj]

            n = new_lik[node_idx]

            if abs(o - n) > 1e-5:
                print(f"Mismatch at node {node_obj}: old={o}, new={n}, diff={abs(o-n)}")
                return False

    print("\n✔ All likelihood values match within tolerance ✔")
    return True