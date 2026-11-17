# Instruments & Resolutions Toolkit

This repository collects the data extraction and modeling work on international legal instruments and United Nations resolutions. 
This corresponds to the Masther's Thesis: Detecting References to International Instruments in UN Resolutions: A Transformer-Based Approach


## Repository structure

| Directory | Description |
|---|---|
| [`document-embeddings/`](./document-embeddings) | Computes embeddings of the resolutions, runs the subject analysis, and retrieves the top-k most similar resolutions for a given topic. |
| [`ilo/`](./ilo) | Instruments from the International Labour Organization (ILO). |
| [`instrument-el/`](./instrument-el) | Prototype of the entity linking engine for instruments. |
| [`instrument-ner/`](./instrument-ner) | Training and evaluation of the named entity recognition (NER) model for instruments. |
| [`ohchr-instruments/`](./ohchr-instruments) | Instruments of the Office of the United Nations High Commissioner for Human Rights (OHCHR). 
| [`qwen3-fine-tune/`](./qwen3-fine-tune) | Prototype of the LLM-based resolution code finder, built by fine-tuning Qwen3. |
| [`spiritrag-corpus/`](./spiritrag-corpus) | Data extractor for the SpiritRAG PDFs. |
| [`unesco_instruments/`](./unesco_instruments) | Instruments of the United Nations Educational, Scientific and Cultural Organization (UNESCO). |
| [`untc-treaties/`](./untc-treaties) | Multilateral treaties from the United Nations Treaty Collection (UNTC). |
| [`wikidata/`](./wikidata) | Instruments from Wikidata, mostly treaties. |