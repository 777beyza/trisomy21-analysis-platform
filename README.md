[https://doi.org/10.5281/zenodo.23100773]
 Traditional Non-Invasive Prenatal Testing (NIPT) for Trisomy 21 (Down 
Syndrome) is typically limited to binary diagnostic outcomes, neglecting the complex, 
systemic nature of the syndrome and its long-term physiological impacts. In this paper, 
a novel, end-to-end Clinical Decision Support System (CDSS) is proposed, which 
integrates machine learning and transcriptomics to provide a holistic framework 
spanning from initial prenatal diagnosis to lifelong predictive monitoring. Initially, the 
platform employs a Random Forest algorithm, reinforced by SHapley Additive 
exPlanations (SHAP) for Explainable AI (XAI), to robustly predict Trisomy 21 risk using 
maternal blood cfDNA parameters (e.g., Chr21 read ratio, fetal fraction, GC bias). For 
high-risk cases, the system triggers a molecular impact mapping module utilizing R
based RNA-Seq differential gene expression (DGE) analysis to monitor the 
overexpression of critical chromosome 21 genes, such as APP and DYRK1A. This 
molecular data feeds into a multi-output predictive model to project the longitudinal 
risk of secondary pathologies, including early-onset Alzheimer's disease and leukemia. 
Furthermore, the platform introduces a pioneering non-invasive monitoring module that 
tracks daily behavioral trends via a relational database and analyzes acoustic 
biomarkers using librosa to detect vocal hesitation and cognitive load—key early 
indicators of dementia. By unifying cfDNA diagnostics, multi-omics risk projection, and 
acoustic biomarker analysis into a single microservice-oriented dashboard, this 
framework transitions Trisomy 21 management from a static diagnostic event to a 
continuous, explainable, and cost-effective predictive continuum. 
