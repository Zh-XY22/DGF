# Third-party sources and redistribution boundaries

- uni2ts: SalesforceAIResearch/uni2ts, vendored with its Apache-2.0 notice in LICENSE.txt. The vendored source and copyright notices are retained with the package. Keep upstream attribution when distributing modifications.
- Moirai public weights: download from the corresponding Salesforce model repository. The model card declares CC-BY-NC-4.0. Weight files are not redistributed here; users must review the model's current terms for their intended use.
- ETT: obtain the CSV files from zhouhaoyi/ETDataset. The repository declares CC-BY-ND-4.0. This candidate supplies local processing code, not a redistributed normalized dataset.
- Selective Learning: algorithm-inspired independent implementation, with the code-consistent AND/absolute-error convention explicitly described. The original repository's applicable code license was not established; its source is not copied into this package.
- verl: historical capped low-variance KL formula is identified in probability.py. The new package implements the continuous-forecast adapter directly and does not require or claim to run RayPPOTrainer.
- PyTorch, GluonTS, Lightning, Hugging Face libraries, NumPy, SciPy, JAX and other dependencies retain their own licenses. The dependency lock identifies installed versions; public installation is preferred over redistribution of the wheel cache.

Sources: https://github.com/SalesforceAIResearch/uni2ts ; https://huggingface.co/Salesforce/moirai-1.0-R-small ; https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE ; https://github.com/GestaltCogTeam/selective-learning ; https://github.com/verl-project/verl .
