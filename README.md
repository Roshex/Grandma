# GRANDMA 👵

**G**ene-tree **R**econciliation **A**lgorithms for **N**ested **D**etection of **M**ultiple **A**llopolyploidizations

GRANDMA is a highly optimized, modern Python reimplementation and massive iterative expansion of the classic GRAMPA algorithm (see https://github.com/gwct/grampa). It reconciles large sets of gene trees against species trees infer polyploid events (e.g., allopolyploid hybridization and WGD) under Maximum Parsimony.

By utilizing flattened array-based tree structures and $O(1)$ Lowest Common Ancestor (LCA) lookups, GRANDMA achieves storage reductions and massive speedups over the original implementation. This efficiency enables the algorithm to move from the realrm of a single-event inference tool into an engine capable of iterative searches across complex evolutionary histories.

## ✨ Key Features

* **Blazing Fast Reconciliation:** A re-engineered core using Range Minimum Query, Euler tours, and integer sparse tables avoids the massive overhead of regex, string parsing, and naive tree traversals. For topological operations, ETE3 is utilized with dedicated wrappers to bypass standard object traversal bottlenecks.
* **Iterative Discovery Modes:** Unlike GRAMPA which is limited to inferring a single event, GRANDMA iteratively discovers multiple nested or independent reticulations using its built-in full, split, or mixed search algorithms.
* **Advanced MUL-Tree Generation:** Go beyond simple diploid hybridization. GRANDMA supports generating MUL-trees with multiple additional clade copies (not just one) and can recursively build them from existing, already-reticulated MUL-trees.
* **Tunable Parsimony Scoring:** Offers finer control over the reconciliation engine by adjusting the specific penalty weights used for calculating the parsimony scores.
* **Biologically Constrained:** Support for a ploidy constraint file. By explicitly defining the maximum allowable genomic copies per species, GRANDMA dynamically prunes the combinatorial search space and prevents biologically impossible reticulations.
* **High Scalability & Minimal Filtration:** Thanks to its massive performance improvements and memory efficiency, GRANDMA can handle massive genomic datasets natively, and requires little-to-no gene tree culling or filtration ('cap' step).
* **Network & eNewick Support:** Pass predefined reticulations via eNewick strings (e.g., `((A,(B)#H1),((C,#H1),D));`) to perform Guided Iterative Searches or directly rank hypotheses.
* **Robust History Tracking & Resumption:** Built for massive cluster environments. GRANDMA continuously logs its iterative history and tree states, allowing you to seamlessly pause, resume, or branch analyses directly from intermediate checkpoints without starting over.
* **Dual Interface:** Run seamlessly from the command line or import it as a Python package for direct integration into Jupyter Notebooks and custom pipelines.

## 📥 Installation

### Option 1: Conda (Recommended)

The easiest way to install GRANDMA and its dependencies is via the Bioconda channel:

```bash
conda install -c bioconda grandma

```

### Option 2: Pip / From Source

To install the latest development version directly from the repository:

```bash
git clone https://github.com/Roshex/grandma.git
cd grandma
pip install .

```

## 🚀 Quick Start (Command Line)

Once installed, GRANDMA is available as a global command.

**Basic Run:**

```bash
grandma -s species_tree.tre -g gene_trees.tre --mode full

```

**Common Arguments:**

* `-s, --spec-tree`: Path to the Rooted, bifurcating species tree.
* `-g, --gene-trees`: Path to the file containing one or more Rooted gene trees.
* `-c, --cap`: Combinatorial cap for ambiguous groups (new default: 15).
* `-m, --mode`: Execution mode (commonly: `single`, `full`, `split`, `mixed`).
* `-h1, --h1` / `-h2, --h2`: Comma-separated list of taxa defining allowed parental lineages to limit the search (used primarily for targeted hypotheses).
* `-x, --ploidy`: Counter-style ploidy constraints input. Limits the maximum number of duplication events allowed for specific clades.
* `-w, --weights`: Penalty weights for duplication and loss events, formatted as 'dup_cost loss_cost' (default: 1 1).
* `-p, --procs`: Number of CPU cores to allocate for parallel multiprocessing.
* `--nestedness`: Strategy for handling nested hybridization events across iterations. Choices: ignore (or i), rectify (or r) - between iterations, model (or m) - during MT generation. Ignore is not advised, and while Model is most accurate, it is also the heaviest mode.
* `--maps`: Outputs the detailed node-to-node orthology mappings for the lowest-scoring MUL-tree to a dedicated file.
* `--plot`: Automatically generates and saves a visualization of the iterative scoring metrics and execution history.


## 🐍 Python API Usage

GRANDMA is designed to be fully accessible from within Python. You can pass the exact same arguments as the CLI and receive the internal data structures (like ETE3 trees and history dictionaries) back for immediate downstream analysis.

Return object is a Python dictionary, with contents that depend on the run parameters. These may include:

1. `final tree`: with 'silt' for the single-labeled, and 'mult' for the multi-labeled representation;
2. `history`: a detailed account of the results of each iterative step (requires iterative option in --mode);
3. `maps`: structure including the reconciliation maps for each iterative step (requires --maps > 0).

```python
from grampack.main import main

# Define your run parameters
args = [
    "-s", "species_tree.tre",
    "-g", "gene_trees.tre",
    "-m", "full",
    "-w", 1, 50,
    "--plot"
]

# Run the pipeline and return objects
results = main(args_list=args, return_objects=True)

# Interact with the results
final_tree = results["final_tree"]["silt"]
final_mult = results["final_tree"]["mult"]
history = results["history"]

print(f"Final Merged Tree: {final_tree.write(format=9)}")

# Inspect the iterative history
for task_id, event in history.items():
    print(f"Task {task_id} best score: {event['score']}")

```

## 📖 Citation

If you use GRANDMA in your research, please cite:

* *Shtein, R. (2026). GRANDMA [Software]*
* *Thomas, G.W.C., Ather, S.H., & Hahn, M.W. (2017). Gene-tree reconciliation with MUL-trees to resolve polyploid events. Systematic Biology 66(6): 1007-1018. https://doi.org/10.1093/sysbio/*
