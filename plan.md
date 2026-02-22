# 🧠 Project Context: GhostWriter (Cross-Platform Live Captions)

## 🎯 Obiettivo del Progetto
Voglio sviluppare "GhostWriter", un'applicazione/demone multipiattaforma (Windows, macOS, Linux) che ascolti l'audio di sistema in uscita (loopback), lo trascriva in tempo reale e mostri i sottotitoli in un overlay trasparente. I testi devono rimanere sempre in primo piano, anche quando apro un gioco a schermo intero (assumendo l'uso della modalità "Borderless/Windowed Fullscreen").

## 🛠️ Stack Tecnologico & Architettura
* **Sistemi Operativi:** macOS (Apple Silicon M-series), Windows 10/11 e Linux.
* **Linguaggio Principale:** Python 3.12+.
* **Cattura Audio (Loopback) - [OS Specifico]:**
  * **macOS:** Utilizzo del driver audio virtuale **BlackHole**.
  * **Windows:** Utilizzo di **WASAPI Loopback**.
  * **Linux:** Utilizzo di **PulseAudio/PipeWire monitor**.
  Lo script deve rilevare automaticamente il sistema operativo (`platform.system()`) e usare il metodo di cattura corretto.
* **Motore STT (Cuore del progetto):**
  * **Modello:** **Distil-Whisper** (es. `distil-large-v3` per multilingua o `distil-medium.en` per inglese).
  * **Libreria:** `faster-whisper` (sfruttando CTranslate2 per la massima velocità).
  * **Hardware:**
      * **macOS:** Sfruttare accelerazione CPU/Metal se supportata da CTranslate2, altrimenti CPU ottimizzata (ARM64).
      * **Windows/Linux:** Sfruttare CUDA (NVIDIA) se disponibile, altrimenti fare fallback su CPU con istruzioni AVX.
* **Interfaccia Grafica (Overlay):** `PyQt6`. La finestra deve essere frameless, con sfondo trasparente, impostata per stare sempre in primo piano (`WindowStaysOnTopHint`) e deve essere completamente "click-through" (ignorare gli input del mouse e della tastiera per non disturbare l'uso del PC o dei giochi).

## ⚠️ Vincoli e Sfide Tecniche
1.  **Latenza Ultra-Bassa:** L'uso di Distil-Whisper è mirato a minimizzare il ritardo. Il codice deve essere ottimizzato per non creare colli di bottiglia tra l'audio e il modello.
2.  **Voice Activity Detection (VAD):** È strettamente necessario un VAD (es. `silero-vad`) per tagliare l'audio nei momenti di silenzio e inviare al modello solo il parlato effettivo.
3.  **Cross-Platform Quirks:** Le API audio cambiano drasticamente tra OS. La logica di cattura deve essere ben astratta.
4.  **Concorrenza:** Gestione rigorosa dei thread e dei processi. Pipeline suggerita: Audio Thread -> VAD/Buffer -> Inference Thread -> Signal -> UI Thread.
5.  **Overlay e Giochi:** Su Windows e Linux, assicurarsi che le flag di PyQt6 forzino il rendering sopra le finestre "Borderless Fullscreen".

## 📋 Roadmap di Sviluppo a Fasi
Ti chiederò di aiutarmi sviluppando un modulo alla volta, senza scrivermi tutto il programma in una volta sola:

* **Fase 1 - Audio Capture Cross-Platform:** Creare uno script che identifichi l'OS e catturi l'audio di sistema (WASAPI, BlackHole o PipeWire). Visualizzazione del livello del volume a console per debug.
* **Fase 2 - Motore Distil-Whisper & VAD:** Integrare `faster-whisper` caricando un modello Distil-Whisper. Integrare `silero-vad`. Testare la trascrizione misurando il "Time to text".
* **Fase 3 - La Pipeline Real-Time:** Unire Fase 1 e Fase 2. Creare un buffer circolare per l'audio + VAD + Distil-Whisper in streaming continuo.
* **Fase 4 - UI Overlay:** Creare la finestra trasparente, always-on-top e click-through con PyQt6 e collegarla all'output testuale della pipeline.

## 🤖 Istruzioni per te (Assistente AI)
* Usa specificamente i modelli **Distil-Whisper** quando scrivi il codice per la parte STT.
* Scrivi codice modulare, tipizzato (type hints) e gestisci accuratamente le eccezioni per ogni OS.
* Analizza e previeni i problemi di concorrenza tra i thread per evitare freeze della UI o del flusso audio.
* Quando ti fornisco questo file di contesto, **rispondimi unicamente confermando che hai compreso l'architettura e chiedendomi se voglio iniziare con la Fase 1.** Non generare codice finché non te lo chiedo esplicitamente.