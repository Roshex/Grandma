from pathlib import Path

from .models import SmrtTree

class OrthologyLabeler:
    @staticmethod
    def run(gene_trees: dict, min_maps: dict, min_tree: SmrtTree, 
            hybrid_clade: list, out_dir: str, prefix: str):
            
        orth_file_path = Path(out_dir) / f"{prefix}_orthologies.txt"
        label_file_path = Path(out_dir) / f"{prefix}_labeled_trees.txt"
        
        with open(orth_file_path, "w") as f_orth, open(label_file_path, "w") as f_lbl:
            
            # Header
            f_orth.write(min_tree.to_string(internal_labels=False))
            f_orth.write("\n------------------\n")
            
            for gene_num, gene_data in gene_trees.items():
                gt_obj = gene_data[0]
                results = min_maps.get(gene_num, [])
                
                if not results: continue
                
                # Check ties
                if len(results) > 1:
                    msg = f"{gene_num}\t* {len(results)} maps tied. Mapping all.\n"
                    f_orth.write(msg)
                    f_lbl.write(msg)
                    
                for res in results:
                    maps, dups = res[3], res[4]
                    
                    # 1. Label Tree String
                    # Reconstruct tree string with +/^ labels based on map
                    # This logic mimics `orthLabel` string replace
                    temp_gt_str = gt_obj.to_string() # Start with clean string
                    
                    spec_genes = {}
                    
                    # Logic to identify Paralog/Homoeolog
                    # This requires re-traversing the gene tree and checking LCAs
                    # It's complex to do on the string, better to do on the object
                    
                    # Identify genes in hybrid clade
                    for node in gt_obj.ete_tree.iter_leaves():
                        sp = node.name.split("_")[-1]
                        if sp in hybrid_clade:
                            cur_map = maps[node.name][0]
                            marker = "+" if "*" in cur_map else "^"
                            
                            # Update string representation (naive replace for compatibility)
                            temp_gt_str = temp_gt_str.replace(node.name, node.name + marker)
                            
                            if sp not in spec_genes: spec_genes[sp] = []
                            spec_genes[sp].append(node.name)
                            
                    f_lbl.write(f"{gene_num}\t{temp_gt_str}\n")
                    
                    # 2. Pairwise Orthology
                    outline = f"{gene_num}\t"
                    done = []
                    
                    for sp, genes in spec_genes.items():
                        if len(genes) == 1:
                            outline += f"{genes[0]}-SINGLE\t"
                        else:
                            for g1 in genes:
                                for g2 in genes:
                                    if g1 == g2: continue
                                    if {g1, g2} in [set(x) for x in done]: continue
                                    done.append([g1, g2])
                                    
                                    # Get LCA of g1, g2 in gene tree
                                    lca_node = gt_obj.ete_tree.get_common_ancestor([
                                        gt_obj.get_node(g1), gt_obj.get_node(g2)
                                    ])
                                    
                                    # Check if LCA is Dup
                                    is_dup = dups.get(lca_node.name, 0)
                                    rel_type = "PARALOG" if is_dup else "HOMOEOLOG"
                                    
                                    outline += f"{g1}-{g2}-{rel_type}\t"
                                    
                    f_orth.write(outline[:-1] + "\n")