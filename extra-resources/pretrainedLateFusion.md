%%{init: {
  "theme": "base",
  "themeVariables": {
    "fontSize": "30px",
    "fontFamily": "trebuchet ms, verdana, arial, sans-serif",
    "lineColor": "#333333"
  },
  "flowchart": {
    "htmlLabels": true,
    "nodeSpacing": 90,
    "rankSpacing": 110,
    "curve": "basis",
    "useMaxWidth": true
  },
  "securityLevel": "loose"
}}%%
flowchart LR
    %% --- Raw input ---
    R["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/rawsignal.png' width='80px' ><br><b>Raw ECGs</b><br>(12 × 1000)"]

    %% --- CWT ---
    CWT["<b>CWT</b><br>(Morlet wavelet)"]

    %% --- Scalogram & Phasogram ---
    SCL["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/scalogram.png' width='80px' ><br><b>Scalograms</b><br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/phasogram.png' width='80px' ><br><b>Phasograms</b><br>(224 × 224 × 12)"]

    %% --- Flow: CWT generates both ---
    R --> CWT
    CWT --> SCL
    CWT --> PHS

    %% --- Dual Adapters ---
    SCL --> A1["<b>Adapter</b><br>1×1 Conv<br>12 → 3 ch"]
    PHS --> A2["<b>Adapter</b><br>1×1 Conv<br>12 → 3 ch"]

    %% --- Dual Pretrained Backbones ---
    A1 --> E1["<b>Pretrained Backbone</b><br>→ fₛ ∈ R^(B×d)"]
    A2 --> E2["<b>Pretrained Backbone</b><br>→ fₚ ∈ R^(B×d)"]

    %% --- Concat Block ---
    E1 --> C["<b>Concat</b><br>[fₛ; fₚ] ∈ R^(B×2d)"]
    E2 --> C

    %% --- Fusion + Classifier ---
    C --> FUS["<b>Fusion FC</b><br>2d → 1024<br>ReLU → BN → Drop(0.3)"]
    FUS --> CLS["<b>Classifier</b><br>1024 → 512 → 5<br>ReLU → BN → Drop(0.3)"]

    %% --- Output ---
    subgraph Output ["<b>Final Multilabel Classifier</b>"]
        direction TB
        CLS --> O1(( ))
        CLS --> O2(( ))
        CLS --> O3(( ))
        CLS --> O4(( ))
        CLS --> O5(( ))
    end


    %% --- Styling ---
    classDef img    fill:#ffffff,stroke:#cccccc,stroke-width:2.5px,color:#000000
    classDef mod    fill:#e3f2fd,stroke:#1565c0,stroke-width:2.5px,color:#000000
    classDef join   fill:#ede7f6,stroke:#5e35b1,stroke-width:2.5px,color:#000000
    classDef head   fill:#e8f5e9,stroke:#2e7d32,stroke-width:2.5px,color:#000000
    classDef neuron fill:#c8e6c9,stroke:#2e7d32,stroke-width:2.5px,color:#000000

    class R,SCL,PHS,CWT img
    class A1,A2,E1,E2,FUS,CLS mod
    class C join
    class SIG head
    class O1,O2,O3,O4,O5 neuron