# Technical Appendix - CT-Based AI-Driven Radiogenomics for Gallbladder Cancer

## Appendix A: Detailed Equipment and Resource Requirements

### A.1 Computing Infrastructure Requirements

| Component | Specification | Quantity | Unit Cost | Total Cost |
|-----------|---------------|----------|-----------|------------|
| **GPU Cluster** | NVIDIA A100 80GB | 8 units | $15,000 | $120,000 |
| **Workstations** | High-performance imaging workstations | 6 units | $8,000 | $48,000 |
| **Storage System** | High-speed NAS with 500TB capacity | 2 units | $25,000 | $50,000 |
| **Network Infrastructure** | 10Gb Ethernet switches and cabling | 1 set | $15,000 | $15,000 |
| **Backup Systems** | Automated backup and disaster recovery | 1 set | $20,000 | $20,000 |
| **Cloud Computing** | AWS/Azure credits for federated learning | 36 months | $3,000/month | $108,000 |

**Total Computing Infrastructure: $361,000**

### A.2 Software and Licensing Requirements

| Software | License Type | Annual Cost | 3-Year Total |
|----------|--------------|-------------|--------------|
| MATLAB with Image Processing Toolbox | Academic license | $2,500 | $7,500 |
| Python ML Libraries (PyTorch, TensorFlow) | Open source | $0 | $0 |
| 3D Slicer Medical Imaging Platform | Open source | $0 | $0 |
| REDCap Database Platform | Institutional license | $5,000 | $15,000 |
| Statistical Software (R, SAS) | Academic license | $3,000 | $9,000 |
| DICOM Viewer and Analysis Software | Commercial license | $8,000 | $24,000 |
| Version Control and Collaboration Tools | Premium subscriptions | $2,000 | $6,000 |

**Total Software Licensing: $61,500**

### A.3 Personnel Effort and Salary Support

| Position | FTE | Annual Salary | 3-Year Total |
|----------|-----|---------------|--------------|
| Principal Investigator | 30% | $180,000 | $162,000 |
| Co-Investigator 1 (Oncology) | 25% | $160,000 | $120,000 |
| Co-Investigator 2 (Biostatistics) | 20% | $140,000 | $84,000 |
| Co-Investigator 3 (Pathology) | 20% | $150,000 | $90,000 |
| Research Scientist 1 (AI/ML) | 50% | $95,000 | $142,500 |
| Research Scientist 2 (Imaging) | 50% | $90,000 | $135,000 |
| Clinical Research Coordinator | 75% | $55,000 | $123,750 |
| Biostatistician | 40% | $75,000 | $90,000 |
| Bioinformatics Analyst | 60% | $80,000 | $144,000 |
| Graduate Student Researchers | 200% | $35,000 | $210,000 |

**Total Personnel Costs (with benefits at 30%): $1,691,325**

## Appendix B: Detailed Technical Specifications

### B.1 CT Imaging Protocol Specifications

#### B.1.1 Scanner Requirements

**Minimum Technical Specifications:**
- Multi-detector CT with ≥64 detector rows
- Dual-energy capability (preferred)
- Iterative reconstruction algorithms
- Sub-millimeter slice reconstruction capability
- Advanced post-processing workstation

**Recommended Vendors and Models:**
- Siemens SOMATOM Force, Edge, or Drive
- GE Revolution CT, Discovery CT750 HD
- Philips iCT, Brilliance BigBore
- Canon Aquilion ONE, Aquilion Prime

#### B.1.2 Acquisition Protocol Details

**Pre-contrast Phase:**
- Tube voltage: 120 kVp
- Tube current: Auto-modulated (reference 200 mAs)
- Rotation time: 0.5-0.75 seconds
- Pitch: 1.2-1.5
- Slice collimation: 64 × 0.625 mm
- Reconstruction thickness: 1.25-2.5 mm
- Reconstruction interval: 1.0-2.0 mm

**Arterial Phase (25-30 seconds post-injection):**
- Contrast agent: Iodinated contrast (300-350 mg I/mL)
- Injection rate: 3-4 mL/second
- Total volume: 100-120 mL
- Saline chaser: 40-50 mL at same rate

**Portal Venous Phase (60-70 seconds post-injection):**
- Same technical parameters as arterial phase
- Optimized for hepatic and vascular enhancement

### B.2 Genomic Profiling Specifications

#### B.2.1 Next-Generation Sequencing Panel

**Target Gene Panel (500+ genes):**

**Tier 1 - Primary Targets (25 genes):**
- KRAS, NRAS, HRAS
- TP53, MDM2, MDM4
- PIK3CA, AKT1, PTEN
- ERBB2, ERBB3, EGFR
- BRCA1, BRCA2, ATM
- IDH1, IDH2, ARID1A
- FGFR2, FGFR3, FGFR4
- STK11, CDKN2A, RB1
- MYC, CCND1

**Tier 2 - Secondary Targets (100+ genes):**
- DNA damage repair genes (PALB2, CHEK2, RAD51C, etc.)
- Cell cycle regulation genes (CCNE1, CDK4, CDK6, etc.)
- Apoptosis pathway genes (BCL2, BAX, BAK1, etc.)
- Immune checkpoint genes (PD-L1, PD-1, CTLA-4, etc.)
- Metabolic pathway genes (MTOR, TSC1, TSC2, etc.)

**Tier 3 - Research Targets (375+ genes):**
- Comprehensive cancer gene panel
- Emerging therapeutic targets
- Resistance mechanisms
- Tumor suppressor genes

#### B.2.2 Quality Control Metrics

**DNA Quality Requirements:**
- Input DNA quantity: ≥50 ng
- DNA integrity number (DIN): ≥3.0
- 260/280 ratio: 1.8-2.0
- 260/230 ratio: >1.5

**Sequencing Quality Metrics:**
- Mean coverage depth: ≥500x
- Uniformity of coverage: ≥80% of targets at 100x
- On-target rate: ≥85%
- Contamination rate: <2%

### B.3 AI Model Architecture Details

#### B.3.1 Deep Learning Network Architecture

```python
# Simplified model architecture pseudocode

class MultiModalRadiogenomicsNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Image feature extraction branch
        self.image_encoder = nn.Sequential(
            Conv3D(1, 64, kernel_size=3, padding=1),
            BatchNorm3D(64),
            ReLU(),
            MaxPool3D(2),
            
            # Residual blocks with attention
            ResidualBlock3D(64, 128),
            AttentionModule3D(128),
            
            ResidualBlock3D(128, 256),
            AttentionModule3D(256),
            
            GlobalAveragePooling3D(),
            Linear(256, 512)
        )
        
        # Clinical data branch
        self.clinical_encoder = nn.Sequential(
            Linear(clinical_features, 128),
            BatchNorm1D(128),
            ReLU(),
            Dropout(0.3),
            
            Linear(128, 256),
            BatchNorm1D(256),
            ReLU(),
            Dropout(0.3),
            
            Linear(256, 128)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            Linear(512 + 128, 256),
            BatchNorm1D(256),
            ReLU(),
            Dropout(0.4),
            
            Linear(256, 128),
            BatchNorm1D(128),
            ReLU(),
            Dropout(0.3)
        )
        
        # Multi-task prediction heads
        self.gene_predictors = nn.ModuleDict({
            'KRAS': Linear(128, 2),
            'TP53': Linear(128, 2),
            'PIK3CA': Linear(128, 2),
            'ERBB2': Linear(128, 2),
            'BRCA1': Linear(128, 2),
            'BRCA2': Linear(128, 2),
            'IDH1': Linear(128, 2),
            'IDH2': Linear(128, 2)
        })
        
        # Uncertainty estimation
        self.uncertainty_head = Linear(128, len(self.gene_predictors))
        
    def forward(self, image, clinical_data):
        # Extract features
        image_features = self.image_encoder(image)
        clinical_features = self.clinical_encoder(clinical_data)
        
        # Fuse modalities
        fused_features = torch.cat([image_features, clinical_features], dim=1)
        fused_representation = self.fusion(fused_features)
        
        # Multi-task predictions
        predictions = {}
        for gene, predictor in self.gene_predictors.items():
            predictions[gene] = predictor(fused_representation)
        
        # Uncertainty estimates
        uncertainty = self.uncertainty_head(fused_representation)
        
        return predictions, uncertainty
```

#### B.3.2 Training Hyperparameters

**Model Training Configuration:**
- Optimizer: AdamW with weight decay 1e-4
- Learning rate: 1e-4 with cosine annealing
- Batch size: 16 (limited by GPU memory)
- Number of epochs: 100-200
- Early stopping: patience=20 epochs
- Loss function: Multi-task focal loss + uncertainty loss

**Data Augmentation:**
- Spatial augmentations: rotation, translation, scaling
- Intensity augmentations: Gaussian noise, brightness/contrast
- Elastic deformations: grid-based warping
- Mixup and CutMix for regularization

**Regularization Techniques:**
- Dropout rates: 0.2-0.5 (layer-specific)
- L2 weight decay: 1e-4
- Batch normalization with momentum 0.9
- Gradient clipping: max norm 1.0

## Appendix C: Validation and Performance Metrics

### C.1 Statistical Analysis Plan

#### C.1.1 Primary Endpoint Analysis

**Primary Endpoints:**
1. AUC-ROC for KRAS mutation prediction
2. AUC-ROC for TP53 mutation prediction
3. AUC-ROC for PIK3CA mutation prediction

**Statistical Hypotheses:**
- H₀: AUC ≤ 0.65 (not clinically useful)
- H₁: AUC > 0.75 (clinically useful)
- α = 0.05, β = 0.20 (power = 80%)

**Sample Size Justification:**
Based on simulation studies assuming:
- Expected AUC = 0.80
- Null hypothesis AUC = 0.65
- Prevalence of mutations: 15-45%
- Required sample size: 800-1000 patients

#### C.1.2 Secondary Endpoint Analysis

**Secondary Endpoints:**
- Sensitivity and specificity for each mutation
- Positive and negative predictive values
- Calibration metrics (Hosmer-Lemeshow test)
- Clinical utility measures (decision curve analysis)

**Subgroup Analyses:**
- Performance by tumor stage (early vs. advanced)
- Performance by tumor location (fundus, body, neck)
- Performance by patient demographics (age, gender, ethnicity)
- Performance by imaging parameters (scanner vendor, protocol)

### C.2 Model Interpretability Framework

#### C.2.1 Explainable AI Methods

**Gradient-based Methods:**
- Gradient-weighted Class Activation Mapping (Grad-CAM)
- Integrated Gradients
- SmoothGrad for noise reduction

**Perturbation-based Methods:**
- LIME (Local Interpretable Model-agnostic Explanations)
- SHAP (SHapley Additive exPlanations)
- Occlusion sensitivity analysis

**Attention Visualization:**
- Spatial attention heatmaps
- Channel attention weights
- Multi-scale attention visualization

#### C.2.2 Clinical Interpretation Guidelines

**Radiologist Review Protocol:**
1. Automated region highlighting by AI model
2. Independent radiologist review and scoring
3. Consensus review for discordant cases
4. Correlation with known imaging patterns

**Clinical Decision Support:**
- Confidence intervals for predictions
- Risk stratification categories
- Recommended follow-up actions
- Integration with clinical guidelines

## Appendix D: Data Management and Quality Assurance

### D.1 Data Management Plan

#### D.1.1 Database Structure

**Primary Database Tables:**
- Patient demographics and clinical data
- Imaging metadata and quality metrics
- Genomic profiling results
- Treatment and outcome data
- AI model predictions and performance

**Data Dictionary:**
- Standardized variable definitions
- Controlled vocabularies (SNOMED CT, ICD-10)
- Data validation rules and constraints
- Version control and change tracking

#### D.1.2 Quality Control Procedures

**Imaging Quality Control:**
- Automated QC metrics calculation
- Manual review checklist
- Inter-reader agreement assessment
- Regular phantom-based calibration

**Genomic Data Quality Control:**
- Sequencing quality metrics monitoring
- Variant calling pipeline validation
- Contamination and ancestry checks
- Regular proficiency testing

### D.2 Regulatory and Compliance Framework

#### D.2.1 IRB and Ethics Compliance

**Multi-Site IRB Strategy:**
- Central IRB for lead institution
- Local IRB approvals for partner sites
- Standardized consent forms and procedures
- Regular safety and data monitoring

**Data Privacy and Security:**
- HIPAA compliance procedures
- De-identification protocols
- Access control and audit trails
- Encryption for data transmission

#### D.2.2 FDA Regulatory Pathway

**Pre-Submission Strategy:**
- Q-Sub meeting for device classification
- Software as Medical Device (SaMD) framework
- Clinical validation requirements
- Quality management system development

**510(k) Submission Timeline:**
- Pre-submission: Month 24
- 510(k) submission: Month 30
- FDA review period: 6 months
- Commercial clearance: Month 36

---

**Appendix Summary:**
This technical appendix provides comprehensive details supporting the main grant proposal, including detailed resource requirements, technical specifications, model architectures, validation frameworks, and regulatory pathways. The information ensures reproducibility and provides reviewers with sufficient technical depth to evaluate the feasibility and rigor of the proposed research.