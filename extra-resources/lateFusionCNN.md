%%{init: {"securityLevel":"loose","flowchart":{"htmlLabels":true}} }%%
flowchart LR
    %% --- Raw input ---
    R["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/rawsignal.png' width='70px' ><br>**Raw ECGs**<br>(12 × 1000)"]

    %% --- CWT ---
    CWT["CWT<br>(Morlet wavelet)"]

    %% --- Branch ---
    SCL["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/scalogram.png' width='70px' ><br>**Scalograms**<br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/phasogram.png' width='70px' ><br>**Phasograms**<br>(224 × 224 × 12)"]

    %% --- Flow ---
    R --> CWT
    CWT --> SCL
    CWT --> PHS

    %% --- Dual Stream CNN ---
    SCL --> CNN_S["CWT2DCNN<br>(12-ch input)<br>→ 1024-d (avg+max pool)"]
    PHS --> CNN_P["CWT2DCNN<br>(12-ch input)<br>→ 1024-d (avg+max pool)"]

    %% --- Late Fusion ---
    CNN_S --> FUS["Late Fusion<br>Concat [fₛ; fₚ] ∈ R^(B×2048)<br>→ 512 → 5<br>ReLU → Drop(0.3)"]
    CNN_P --> FUS

    %% --- Output ---
    subgraph Output ["Final Multilabel Classifier"]
        direction TB
        FUS --> O1(( ))
        FUS --> O2(( ))
        FUS --> O3(( ))
        FUS --> O4(( ))
        FUS --> O5(( ))
    end

    Output --> SIG["Sigmoid Focal Loss<br>(γ=2, α=0.25)"]

    %% --- Styling ---
    classDef img fill:#ffffff,stroke:#cccccc,stroke-width:1px
    classDef mod fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    classDef head fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef neuron fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px

    class R,SCL,PHS,CWT img
    class CNN_S,CNN_P,FUS mod
    class SIG head
    class O1,O2,O3,O4,O5 neuron