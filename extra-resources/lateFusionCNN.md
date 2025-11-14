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

    %% --- Branch ---
    SCL["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/scalogram.png' width='80px' ><br><b>Scalograms</b><br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/phasogram.png' width='80px' ><br><b>Phasograms</b><br>(224 × 224 × 12)"]

    %% --- Flow ---
    R --> CWT
    CWT --> SCL
    CWT --> PHS

    %% --- Dual Stream CNN ---
    SCL --> CNN_S["<b>CWT2DCNN</b><br>(12-ch input)<br>→ 1024-d (avg+max pool)"]
    PHS --> CNN_P["<b>CWT2DCNN</b><br>(12-ch input)<br>→ 1024-d (avg+max pool)"]

    %% --- Concat Block ---
    CNN_S --> CONCAT["<b>Concat</b><br>[fₛ; fₚ] ∈ R^(B×2048)"]
    CNN_P --> CONCAT

    %% --- Fusion FC ---
    CONCAT --> FUS["<b>Fusion FC</b><br>2048 → 512<br>ReLU → Drop(0.3)"]

    %% --- Classifier ---
    FUS --> CLS["<b>Classifier</b><br>512 → 5"]

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
    class CNN_S,CNN_P,FUS,CLS mod
    class CONCAT join
    class SIG head
    class O1,O2,O3,O4,O5 neuron