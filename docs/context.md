# Context from the originating conversation

Source: [Conformer Ensemble Metric — shared ChatGPT conversation](https://chatgpt.com/share/6ac06666-31e8-83e9-a0e3-eb24fc125fa3).

This document captures the conversation's objectives and evolving recommendations.
It is project background, not a new implementation request or a verified literature
review. The README describes the package's actual behavior.

## Scientific objective

The user is interested primarily in structural diversity: **how different are the
conformers a search discovers**, rather than whether it finds only the lowest
energy conformer. Compare exploration performance separately from energy
performance. GOAT, CREST, and agentic exploration were mentioned as possible
benchmark methods, not a fixed list of required integrations.

The conversation began with ensemble similarity, then shifted to intrinsic
diversity and coverage without declaring a reference ensemble to be ground truth.
Reference-dependent COV/MAT and distribution comparisons such as Wasserstein
distance answer different questions.

## Primary diversity measure

For N conformers of the same molecule, construct a symmetric distance matrix and
average its upper triangle:

\[
D = \frac{2}{N(N-1)}\sum_{i<j}d(C_i,C_j).
\]

This is the mean distance between distinct conformer indices. The chat's example
has four conformers and distances 0.2, 1.1, 2.0, 1.0, 1.9, and 1.2 Å, giving
7.4/6 ≈ 1.23 Å. Redundant conformers contribute small or zero distances.
Report the pairwise-distance distribution alongside the mean: similar averages
can hide very different arrangements of conformational basins.

The discussion favored Pracht's permutation-invariant RMSD (iRMSD) for comparing
methods on the **same molecule**, to address symmetry-equivalent atoms and atom
permutations that can inflate ordinary RMSD. The referenced paper is
[Conformational Pruning via the Permutation Invariant Root-Mean-Square Deviation
of Atomic Positions](https://doi.org/10.1021/acs.jcim.4c02143).

An intermediate recommendation favored iRMSD alone. The later recommendation
reintroduced Torsion Fingerprint Deviation (TFD) as a normalized alternative for
aggregating across molecules with different sizes and flexibilities, with iRMSD
as a complementary score. A normalized distance still needs a stated benchmark
and aggregation policy.

## Ensemble size and molecular flexibility

More returned conformers do not automatically imply greater diversity. Under iid
sampling from the same distribution, mean pairwise distance has the same
expectation at different sample sizes; more samples can improve precision and
discover rare basins. Search depth and search budget still matter.

The chat proposed repeated equal-size subsampling without replacement, using
several common sizes to compare methods. It also proposed diversity and coverage
versus **computational budget** to study exploration efficiency. Subsampling a
finished ensemble and collecting search checkpoints are different procedures.

Raw counts of distinct conformers strongly depend on sampling depth, resolution,
and molecular flexibility. Avoid ranking different molecules using these counts.
RMSD already divides by atom count; the chat corrected the suggestion that RMSD
is literally extensive, while recognizing its remaining molecule dependence.

## Coverage at a structural cutoff

The chat uses “resolution” to mean a distance cutoff δ: nearby structures are
treated as redundant. It proposes N_unique(δ), the retained conformer/cluster
count across multiple cutoffs, as a coverage diagnostic. Later recommendations
explicitly make this secondary to mean pairwise diversity.

Pairwise closeness is not transitive. If A–B and B–C are within δ, A–C can exceed
δ, so the clustering or pruning policy must be explicit. The chat suggested
Pracht-style pruning; the current package instead explicitly defines N_unique
using single-linkage connected components. These definitions are not equivalent.

An intermediate proposed scalar was fractional coverage AUC:

\[
\frac{1}{\delta_{\max}-\delta_{\min}}
\int_{\delta_{\min}}^{\delta_{\max}}
\frac{N_{\mathrm{unique}}(\delta)}{N}\,d\delta.
\]

In the existing API, `curve.auc(normalize=True)` divides by the cutoff span only;
divide this result by N as well to obtain the proposed fractional form. The API
uses trapezoidal interpolation on the supplied grid. AUC is a secondary
diagnostic with an explicit cutoff range and clustering definition.

## Energy relevance and population diversity

The discussion warns that geometrically unusual, very high-energy structures can
inflate a structural score without representing relevant conformations. It
suggests selecting an energy window before analysis or using meaningful
Boltzmann/free-energy populations. Example windows in the chat are illustrative,
not universal thresholds.

With normalized populations p_i, weighted diversity is

\[
Q = \sum_{i,j}p_i p_j d_{ij}, \qquad \sum_i p_i=1.
\]

It describes independent draws from the population, including the possibility of
the same conformer twice. The unweighted mean describes distinct returned indices.
For equal populations, Q = (N−1)D/N, so they differ at finite N. Meaningful physical
populations and search output frequencies answer different scientific questions.

The chat also proposes the complementary effective population count

\[
N_{\mathrm{eff}}=\exp\left(-\sum_{i:p_i>0}p_i\ln p_i\right).
\]

Keep best-energy discovery, geometric diversity, structural coverage, and
population concentration as separate reported quantities.

## Package intent

The final user request in the shared conversation was for a **very lean Python
package**. The stated scope included minimal dependencies, pairwise diversity,
weighted diversity, rarefaction/subsampling, N_unique, curves/AUC, tests, and a
concise README. A pluggable distance callable was chosen to support Pracht-style
iRMSD without inventing an uncertain reimplementation of that algorithm.

The working directory already implements the core scope with NumPy, a callable
distance interface, and an optional upstream iRMSD adapter. Energy-window helpers,
effective population count, a built-in TFD adapter, and search-budget collection
were discussed as context; this document does not make them new requirements.
