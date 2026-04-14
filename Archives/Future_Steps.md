# Future Steps

1. **Test E2E Scenarios with Dashboard UI**: Launch the streamlit dashboard to verify that visual representations (Nodes, Gini Score maps) reflect `output/final_optimal_layout.json`.
2. **Refine Subsystem Interfaces**: Ensure the shapes of demand tensors seamlessly cross the boundary from PyTorch Generation -> Risk GA -> Geospatial representation. If padding is required, consider re-orienting the datasets to conform exactly to IEEE 33-Bus configuration explicitly.
3. **Hardware Acceleration**: Implement tensorized versions of the physics constraints in PyTorch for use on GPUs directly under GA to speed up evaluating thousands of risk variants.
4. **Data Contract Compliance**: Update `ROADMAP.md` checklist marks for "Integration" since they are now implemented.
5. **[COMPLETED] Colab Training Notebook (Lochan Integration)**: Rebuilt `EVolvAI_Training.ipynb` as a fully self-contained Colab notebook. It clones the repo, runs a pure-Python port of Lochan's `GenerateRandomSchedule_new_ScenerioGenerator.m` MATLAB EV scenario generator, builds the entire dataset in RAM, trains the physics-informed TCN-VAE, and downloads results. No MATLAB license required.
6. **Integrate Traffic Flow Data**: 
    - **Step 1:** Define the geographic bounding box (e.g., specific city or county) matching the grid topology.
    - **Step 2:** Decide on the data abstraction level (hourly aggregate volume on main arteries vs node-level estimates).
    - **Step 3:** Source data via OpenStreetMap (OSMnx in Python for road networks) and state DOT APIs / US Census LEHD origin-destination data for flow volumes.
    - **Step 4:** Write a pre-processing pipeline script (`data_pipeline/traffic_preprocess.py`) to map hourly traffic volumes mathematically to EV multiplier nodes in the `C` condition vector.
13. **Algorithm Benchmarking**: Evaluate **Particle Swarm Optimization (PSO)** against the current Genetic Algorithm (GA) for placement optimization to see if "Global Empirical Range Embedding" (GERE) concepts improve convergence speed.
14. **Refine Output Bounding**: Implement hard-clamping logic in the generation script to complement the soft penalties in `physics_loss.py`, ensuring 100% of generated scenarios are physically bounded before handoff.
15. **Train the Gen-Core (GCD-VAE)**: Leverage the newly extracted `bootstrap.py` script to generate 5,000 ACN-based scenarios across 32 nodes over traffic indices to effectively train the GCD-VAE without starving the model.
