**# 🧠 StudyMind AI**

A fully local, privacy-first AI study assistant. Upload your PDFs, ask questions, generate summaries, flashcards, quizzes — all running on your own machine via Ollama + Qwen2.5.

\---

**## Project Structure**

\`\`\`

studymind/

├── app.py              # Streamlit UI (entry point)

├── config.py           # Centralised configuration

├── pdf\_processor.py    # PDF extraction + chunking

├── vector\_store.py     # ChromaDB wrapper

├── rag\_engine.py       # Ollama RAG functions

├── pdf\_exporter.py     # Markdown → PDF via fpdf2

├── requirements.txt

└── README.md

\`\`\`

\---

**## Prerequisites**

\| Requirement | Version | Install |

\|-------------|---------|---------|

\| Python | ≥ 3.10 | [python.org]\([https://python.org](https://python.org)) |

\| Ollama | latest | [ollama.ai]\([https://ollama.ai](https://ollama.ai)) |

\---

**## Setup — Step by Step**

**### 1. Clone / create the project folder**

\`\`\`bash

mkdir studymind && cd studymind

\# (copy all project files here)

\`\`\`

**### 2. Create and activate a Python virtual environment**

\`\`\`bash

python -m venv .venv

\# macOS / Linux:

source .venv/bin/activate

\# Windows (PowerShell):

.venv\Scripts\Activate.ps1

\`\`\`

**### 3. Install Python dependencies**

\`\`\`bash

pip install --upgrade pip

pip install -r requirements.txt

\`\`\`

\> ⏳ First-time install downloads \`sentence-transformers\` (\~500 MB) and the

\> \`all-MiniLM-L6-v2\` embedding model (\~90 MB). Subsequent runs are instant.

**### 4. Install Ollama and pull Qwen2.5**

\`\`\`bash

\# Install Ollama (macOS/Linux one-liner):

curl -fsSL [https://ollama.ai/install.sh](https://ollama.ai/install.sh) | sh

\# Windows: download the installer from [https://ollama.ai/download](https://ollama.ai/download)

\# Pull the model (requires \~4.7 GB disk space for the 7B Q4 variant):

ollama pull qwen2.5

\# Verify Ollama is running:

ollama list

\`\`\`

**### 5. Start Ollama (keep this terminal open)**

\`\`\`bash

ollama serve

\`\`\`

\> Ollama listens on \`[http://localhost:11434](http://localhost:11434)\` by default.

**### 6. Launch StudyMind AI**

Open a **\*\*new terminal\*\*** in the project folder (with the venv active):

\`\`\`bash

streamlit run app.py

\`\`\`

The app will open automatically at **\*\*http\://localhost:8501\*\***.

\---

**## Usage Guide**

**### Indexing PDFs**

1\. Open the sidebar (click **\*\*>\*\*** if collapsed).

2\. Drag and drop one or more PDF files into the uploader.

3\. Click **\*\*⚡ Index Uploaded PDFs\*\***.

4\. Watch the progress bar — the sidebar shows the total chunk count once done.

**### Chat (Tab 1)**

\- Type any question in the input box and press **\*\*Send\*\***.

\- The model streams its answer in real time.

\- Expand **\*\*🔍 Retrieved Context Sources\*\*** to see exactly which page and passage was used.

**### Quick Tools (Tab 2)**

\| Button | What it generates |

\|--------|-------------------|

\| Generate Summary | Structured overview with key concepts |

\| Create Flashcards | 15 Q&A pairs for memorisation |

\| Build Quiz | 10-question multiple-choice exam with answer key |

**### PDF Export (Tab 3)**

1\. Generate content in Quick Tools, or paste your own Markdown.

2\. Set the document title.

3\. Click **\*\*🖨 Generate PDF\*\*** then **\*\*📥 Download PDF\*\***.

\---

**## Configuration**

Edit \`config.py\` to change defaults without touching other files:

\`\`\`python

OLLAMA\_HOST   = "[http://localhost:11434](http://localhost:11434)"  # change if Ollama is on another machine

DEFAULT\_MODEL = "qwen2.5"                 # swap for llama3, mistral, etc.

CHUNK\_SIZE    = 1000                      # characters per chunk

CHUNK\_OVERLAP = 200                       # overlap between chunks

TOP\_K\_RESULTS = 5                         # retrieved chunks per query

\`\`\`

\---

**## Troubleshooting**

\| Symptom | Fix |

\|---------|-----|

\| 🔴 Ollama Offline | Run \`ollama serve\` in a terminal |

\| Model not found | Run \`ollama pull qwen2.5\` |

\| PDF yields no text | PDF is scanned/image-based — use OCR first (e.g., Adobe Acrobat) |

\| Slow responses | Use a smaller model: \`ollama pull qwen2.5:3b\` and select it in Settings |

\| ChromaDB errors | Delete the \`./chroma\_db\` folder and re-index |

\| Port 8501 busy | Run \`streamlit run app.py --server.port 8502\` |  generate and updated readme.md text so when i upload my project to github users would have any problem when reading through the readme.md file also system configuration for all os 