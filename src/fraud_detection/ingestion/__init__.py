"""Deterministic PaySim ingestion package.

Import services from their defining modules. Keeping package initialization side-effect free lets
the read-only API use shared hashing/contracts without loading data-frame or training dependencies.
"""
