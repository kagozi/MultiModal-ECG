flowchart LR
    %% --- Raw input ---
    R["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/rawsignal.png' width='70px' ><br>Raw ECGs<br>(12 × 1000)"]

    %% --- CWT (single block) ---
    CWT["CWT<br>(Morlet wavelet)"]

    %% --- Branch to Scalogram & Phasogram ---
    SCL["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources//scalogram.png' width='70px' ><br>Scalograms<br>(224 × 224 × 12)"]
    PHS["<img src='https://raw.githubusercontent.com/kagozi/MultiModal-ECG/main/extra-resources/phasogram.png' width='70px' ><br>Phasograms<br>(224 × 224 × 12)"]

    %% --- Flow ---
    R --> CWT
    CWT --> SCL
    CWT --> PHS

    %% --- Choose ONE branch (user picks) ---
    subgraph Input_Choice ["Input Stream (choose one)"]
        direction TB
        SCL
        PHS
    end

    %% --- OR (user selection) ---
    SCL --> OR["OR"]
    PHS --> OR

    %% --- Model (unchanged) ---
    OR --> PT["2D CNN<br>(224, 224, 12)<br>→ f ∈ ℝ^(B×d)"]
    PT --> CLS["Classifier<br>d → 512 → 5<br>ReLU → BN → Drop"]

    %% --- Output ---
    subgraph Output ["Final Multilabel Classifier"]
        direction TB
        CLS --> O1(( ))
        CLS --> O2(( ))
        CLS --> O3(( ))
        CLS --> O4(( ))
        CLS --> O5(( ))
    end

    Output --> SIG["Sigmoid Focal Loss<br>(γ=2, α=0.25)"]

    %% --- Styling ---
    classDef img fill:#ffffff,stroke:#cccccc,stroke-width:1px
    classDef mod fill:#e3f2fd,stroke:#1565c0,stroke-width:1px
    classDef head fill:#e8f5e9,stroke:#2e7d32,stroke-width:1px
    classDef neuron fill:#c8e6c9,stroke:#2e7d32,stroke-width:1px

    class R,SCL,PHS,CWT img
    class PT,OR mod
    class CLS,SIG head
    class O1,O2,O3,O4,O5 neuron