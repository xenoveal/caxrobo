"""
Mutator plug-ins (v0.3.0 Phase 6): the legal moves a population search may make.

A sub-package so framework.registry.load_all()'s pkgutil.walk_packages reaches
the modules below and their decorators run. Every mutator here is a plain
function taking (graph, rng, **params) and returning a NEW, VALIDATED
StrategyGraph — never a mutation of its input, because the parent graph is reused
for elitism, lineage and dedup, and an in-place edit would destroy
reproducibility with no test obviously failing.
"""
