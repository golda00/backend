# 🗺️ Telecom Vision API

> An AI-powered backend service for automated analysis of Telecom network maps (Coax & Fiber). It compares Before/After maps, detects infrastructure changes using computer vision, generates annotated PDF reports with callouts, and stamps Survey information onto maps — all served through a clean REST API.

---

## 📌 Table of Contents
- [What This Project Does](#-what-this-project-does)
- [Architecture Overview](#-architecture-overview)
- [Tech Stack](#-tech-stack)
- [What is Celery and Why We Use It](#-what-is-celery-and-why-we-use-it)
- [Project Structure](#-project-structure)
- [Prerequisites & Setup](#-prerequisites--setup)
- [Environment Variables](#-environment-variables)
- [Running the Application](#-running-the-application)
- [API Endpoints](#-api-endpoints)
  - [Coax Map Analysis](#coax-map-analysis-post-apiv1jobs)
  - [Fiber Overview Analysis](#fiber-overview-analysis-post-apiv1jobsfiber-overview)
  - [Fiber Overview Before Map](#fiber-overview-before-map-post-apiv1jobsfiber-overview-before)
  - [Coax Before Map](#coax-before-map-post-apiv1jobscoax-before)
  - [Job Polling](#job-polling)
- [Job Lifecycle (How a Job Works)](#-job-lifecycle)
- [Map Analysis Pipelines](#-map-analysis-pipelines)
  - [Coax Map Pipeline](#1-coax-map-analysis-pipeline)
  - [Fiber Overview Pipeline](#2-fiber-overview-pipeline)
  - [Before Map Pipelines (Stateless)](#3-before-map-pipelines-stateless)
- [AI Models](#-ai-models)
- [PDF Reporting & Callouts](#-pdf-reporting--callouts)
- [Model Weights](#-model-weights)

---

## 🔍 What This Project Does

Telecom network engineers regularly compare "Before" and "After" maps of cable installations to identify what needs to be changed or upgraded. This API automates that comparison:

1. **Upload** a Before PDF + After PDF (Coax analysis) **or** a single Fiber Overview map.
2. The backend **detects objects** (amplifiers, taps, power supplies, nodes, splitters, etc.) in both maps using trained YOLO models.
3. Objects are **matched** across maps using a 4-pass spatial algorithm.
4. A **rule engine** evaluates each matched pair and generates descriptive **callout labels** (e.g., "UPGRADE POWER SUPPLY", "WARNING: Power Supply Detected").
5. Callouts are **printed onto the original PDF** as vector annotations with arrows pointing to each detected object.
6. An optional **Survey Info block** (image + text box with PrismID, Node Name, Instance, page count) is stamped into the top-right corner of the map.
7. The annotated PDF is available to download via the API.

---

## 🏗️ Architecture Overview

```
Client (e.g. Postman / Frontend)
        │
        ▼
┌─────────────────────┐
│    FastAPI API       │  ← Handles HTTP, validates files, stores job in Redis, returns job_id
│   (uvicorn :8000)   │
└────────┬────────────┘
         │  .delay(job_id)   (fire-and-forget)
         ▼
┌─────────────────────┐
│  Redis (Job Queue)  │  ← Acts as message broker AND job state store
│  localhost:6379      │
└────────┬────────────┘
         │  task picked up
         ▼
┌─────────────────────┐
│   Celery Worker     │  ← Runs the heavy processing (AI inference, PDF generation)
│  (thread pool x3)   │
└─────────────────────┘
         │  writes report PDF
         ▼
     storage/outputs/
```

The API returns instantly with a `job_id`. The client polls `GET /api/v1/jobs/{job_id}` to track progress. When the job is `completed`, the report PDF can be downloaded.

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **API Framework** | [FastAPI](https://fastapi.tiangolo.com/) | Async, typed, auto-generates Swagger docs |
| **ASGI Server** | [Uvicorn](https://www.uvicorn.org/) | High-performance Python async server |
| **Task Queue** | [Celery](https://docs.celeryq.dev/) | Offloads slow AI jobs from the API process |
| **Message Broker** | [Redis](https://redis.io/) | Celery's queue + job state storage |
| **Object Detection** | [YOLOv8 (Ultralytics)](https://docs.ultralytics.com/) | Fast, accurate real-time object detector |
| **PDF Processing** | [PyMuPDF (fitz)](https://pymupdf.readthedocs.io/) | Read, write and annotate PDFs |
| **Image Processing** | [OpenCV](https://opencv.org/) | Rasterize PDFs, image alignment, tile processing |
| **OCR** | [EasyOCR](https://github.com/JaidedAI/EasyOCR) | Reads text values inside detected objects (tap values, voltage etc.) |
| **Configuration** | [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) | .env-based typed settings |
| **Rate Limiting** | [SlowAPI](https://github.com/laurentin/slowapi) | Prevents API abuse |

---

## ⚙️ What is Celery and Why We Use It

### The Problem
Running AI inference (YOLO detection) on large, high-resolution PDFs takes **30 seconds to several minutes** of CPU/GPU time. If the FastAPI web server had to do this directly inside the HTTP request handler, the client would have to wait the whole time with an open connection. This is bad for:
- Network timeouts
- Server concurrency (one stuck request blocks others)
- User experience

### The Solution — Celery
**Celery** is a distributed task queue system. Think of it as a **post office for work**:

1. The API **drops a task envelope** (containing the `job_id`) into Redis (the "mailbox").
2. The API **immediately returns** a 202 Accepted response to the client with the `job_id`.
3. A **Celery Worker process** (running separately) picks up the envelope from Redis, opens it, and executes the heavy computation.
4. The worker writes progress/status back to Redis as it runs.
5. The client **polls** `GET /jobs/{job_id}` to check progress — the API simply reads from Redis and returns the status instantly.

```
POST /jobs  →  API stores job in Redis  →  returns job_id  (< 100ms)
                                    ↓
                          Celery Worker starts
                          processing in background
                                    ↓
Repeated  GET /jobs/{id}  →  reads status from Redis
                                    ↓
                          Worker finishes, sets status=completed
                                    ↓
          GET /jobs/{id}/download  →  returns annotated PDF
```

In this project Celery uses `--pool=threads` (thread pool, not processes) because YOLO model loading is expensive — each thread shares the already-loaded model.

---

## 📁 Project Structure

```
telecom-vision-api/
├── app/
│   ├── main.py                  # FastAPI app factory, middleware, startup
│   ├── core/
│   │   ├── config.py            # All settings (from .env)
│   │   ├── logging.py           # Logging setup
│   │   └── store.py             # Redis job store (RedisJobStore)
│   ├── api/
│   │   └── v1/
│   │       ├── jobs.py          # All job submission endpoints
│   │       └── health.py        # Health check endpoint
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response schemas
│   ├── services/
│   │   ├── vision.py            # YOLOv8 detection wrapper (CoaxVisionDetector)
│   │   ├── alignment.py         # PDF→image conversion, tile stitching, homography
│   │   ├── matching.py          # 4-pass before/after object matching
│   │   ├── rules.py             # Rule engine → generates callout labels
│   │   ├── reporting.py         # PDF annotation (callouts + survey info overlay)
│   │   ├── fiber_overview.py    # FiberOverviewProcessor (YOLO + skeletonization)
│   │   ├── fiber_before.py      # Fiber before-map stamper (no AI)
│   │   └── coax_before.py       # Coax before-map stamper (smart corner detect, no AI)
│   └── workers/
│       ├── celery_app.py        # Celery app configuration
│       ├── tasks.py             # Celery task definitions
│       └── pipeline.py          # Synchronous pipeline logic for all map types
├── model_weights/
│   ├── best.pt                  # Main Coax object detector
│   ├── power_supply_best.pt     # Power supply detector
│   ├── 3x3_4x4_new_model.pt     # Node type classifier
│   ├── Internal_best.pt         # Internal splitter detector
│   └── fiber_node_model.pt      # Fiber node detector
├── storage/
│   ├── uploads/                 # Uploaded PDFs (per job_id folder)
│   └── outputs/                 # Generated annotated PDFs
├── .env                         # Your environment variables (gitignored)
├── .env.example                 # Template for .env
└── requirements.txt             # Python dependencies
```

---

## 🔧 Prerequisites & Setup

### 1. Install Redis
Redis is required for the job queue. On Windows the easiest way is via [Memurai](https://www.memurai.com/) or WSL.

```bash
# Test Redis is running:
redis-cli ping
# Should return: PONG
```

### 2. Create Python Environment
```bash
conda create -n telecom python=3.10
conda activate telecom
pip install -r requirements.txt
```

### 3. Place Model Weights
Download all `.pt` files and place them in the `model_weights/` folder:
- `best.pt`
- `power_supply_best.pt`
- `3x3_4x4_new_model.pt`
- `Internal_best.pt`
- `fiber_node_model.pt`

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env and set REDIS_URL etc. as needed
```

---

## 🌍 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `PDF_DPI` | `300` | PDF rendering resolution |
| `TILE_SIZE` | `640` | YOLO tile size in pixels |
| `TILE_OVERLAP` | `0.2` | Tile overlap fraction (20%) |
| `USE_GPU` | `True` | Use GPU for YOLO inference |
| `MAX_UPLOAD_BYTES` | `209715200` | Max upload file size (200 MB) |
| `JOB_TIMEOUT_SECONDS` | `1800` | Max pipeline runtime (30 min) |
| `JOB_RETENTION_HOURS` | `24` | Hours until old jobs are auto-deleted |
| `PIPELINE_WORKERS` | `4` | Thread pool size for pipelines |

---

## ▶️ Running the Application

You need **three** separate terminal windows:

**Terminal 1 — Redis** (if not already running as a service):
```bash
redis-server
```

**Terminal 2 — FastAPI Server:**
```bash
cd telecom-vision-api
conda activate telecom
uvicorn app.main:app --reload --port 8000
```

**Terminal 3 — Celery Worker:**
```bash
cd telecom-vision-api
conda activate telecom
celery -A app.workers.celery_app worker --loglevel=info --pool=threads --concurrency=3
```

Open Swagger UI at: **http://localhost:8000/docs**

---

## 🌐 API Endpoints

All endpoints are prefixed with `/api/v1`.

---

### Coax Map Analysis `POST /api/v1/jobs`

Upload a **Before** and **After** Coax map PDF for full AI-powered change detection.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `before_pdf` | file | ✅ | Before map PDF |
| `after_pdf` | file | ✅ | After map PDF |
| `survey_image` | file | ❌ | Survey Info screenshot |
| `prism_id` | string | ❌ | e.g. `4147677_4147697` |
| `map_type` | string | ❌ | e.g. `AFTER` |
| `node_name` | string | ❌ | e.g. `OX003A_OX003B` |
| `instance` | string | ❌ | e.g. `1` |
| `dpi` | int | ❌ | Default: 300 |

**Response:** `202 Accepted` → `{ "job_id": "...", "job_token": "..." }`

---

### Fiber Overview Analysis `POST /api/v1/jobs/fiber-overview`

Upload a single **Fiber Overview** map PDF for node detection, cable tracing, and port detection.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | ✅ | Fiber Overview PDF |
| `survey_image` | file | ❌ | Survey Info screenshot |
| `prism_id` | string | ❌ | Prism ID |
| `node_name` | string | ❌ | Node name |
| `instance` | string | ❌ | Instance |
| `is_connected` | bool | ❌ | Is node connected to hub? Default: True |
| `hub_name` | string | ❌ | Hub name (if connected) |
| `port_name` | string | ❌ | Port/panel name (if connected) |
| `splice_can_name` | string | ❌ | Splice can name (if not connected) |

---

### Fiber Overview Before Map `POST /api/v1/jobs/fiber-overview-before`

Stamps Survey Info + Title Box onto a **Fiber Overview BEFORE** map. No AI inference.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `before_pdf` | file | ✅ | Fiber Overview Before PDF |
| `survey_image` | file | ❌ | Survey Info screenshot |
| `prism_id` | string | ❌ | Prism ID |
| `node_name` | string | ❌ | Node name |
| `instance` | string | ❌ | Instance |
| `map_type` | string | ❌ | Default: `BEFORE` |

---

### Coax Before Map `POST /api/v1/jobs/coax-before`

Stamps Survey Info + Title Box onto a **Coax BEFORE** map using smart whitest corner detection. No AI inference.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `before_pdf` | file | ✅ | Coax Before PDF |
| `survey_image` | file | ❌ | Survey Info screenshot |
| `prism_id` | string | ❌ | Prism ID |
| `node_name` | string | ❌ | Node name |
| `instance` | string | ❌ | Instance |
| `map_type` | string | ❌ | Default: `BEFORE PRINT` |

---

### Job Polling

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/v1/jobs/{job_id}` | Poll job status and progress % |
| `GET` | `/api/v1/jobs/{job_id}/result` | Get result + callout list (when completed) |
| `GET` | `/api/v1/jobs/{job_id}/download` | Download annotated PDF |

> ⚠️ All requests after job creation require the `X-Job-Token` header (returned in the job creation response).

**Job Status Values:**
`queued` → `processing` → `reporting` → `completed` (or `failed`)

---

## 🔄 Job Lifecycle

```
Client POSTs files
        │
        ▼
API validates PDFs (magic bytes + extension)
        │
        ▼
Files saved to storage/uploads/{job_id}/
        │
        ▼
Job record written to Redis with status = "queued"
        │
        ▼
202 Accepted returned to client (with job_id + job_token)
        │
        ▼ (background)
Celery Worker picks up job_id from Redis queue
        │
        ▼
Pipeline runs (10% → 30% → 50% → 80% → 100%)
Each stage updates Redis
        │
        ▼
Annotated PDF saved to storage/outputs/{job_id}/report.pdf
        │
        ▼
Redis record updated: status = "completed", report_path = "..."
        │
        ▼
Client GETs /download → receives annotated PDF
```

---

## 🗺️ Map Analysis Pipelines

### 1. Coax Map Analysis Pipeline

**File:** `app/workers/pipeline.py` → `run_pipeline_sync()`

1. **PDF → High-res Image** (OpenCV, PyMuPDF)
2. **Alignment** — Before and After maps are aligned using feature matching (SIFT/ORB) to produce a homography matrix `W_inv`. This corrects for any rotation, scale or translation differences between the two scans.
3. **Tiling** — Images are cut into overlapping 640×640 tiles for YOLO inference.
4. **Object Detection** — Each tile is run through 4 YOLO models in parallel:
   - `best.pt` — Main symbol detector (amplifiers, taps, splitters, nodes, terminators etc.)
   - `power_supply_best.pt` — Specialized power supply detector
   - `3x3_4x4_new_model.pt` — Node type classifier
   - `Internal_best.pt` — Internal splitter detector
5. **OCR** — Text values inside detected objects are read using EasyOCR (e.g., tap values "17", voltages "90V").
6. **Matching** (`app/services/matching.py`) — Detected objects in Before and After maps are matched using 4 passes:
   - Pass 1: Strict IoU + same class
   - Pass 2: IoU only (allows class change, e.g., LE → Amplifier)
   - Pass 3: Proximity + same class
   - Pass 4: Proximity, any class
7. **Rule Engine** (`app/services/rules.py`) — Generates callout text for each change:
   - `A` = Amplifier present
   - `B` = Line Extender → Amplifier upgrade
   - `E` = Tap value change
   - `G` = Splitter type change
   - `H` = New LE/Booster added
   - `J` = Equalizer removed
   - `UPGRADE POWER SUPPLY` = Power supply voltage changed
   - `WARNING: Power Supply Detected` = Power supply present
   - `UPGRADE NODE` = Node upgraded, etc.
8. **PDF Report** — Callouts are rendered as vector annotations with arrows pointing at detected objects. Duplicate callouts within 50 PDF points are automatically merged.
9. **Survey Info Overlay** — Survey image + Title Box stamped in top-right corner.

---

### 2. Fiber Overview Pipeline

**File:** `app/workers/pipeline.py` → `run_fiber_overview_pipeline()`  
**Service:** `app/services/fiber_overview.py` → `FiberOverviewProcessor`

1. PDF → Image
2. **Node Detection** — `FiberOverviewProcessor` uses `fiber_node_model.pt` to locate the fiber node in the map.
3. **Cable Tracing** — Skeletonization (scikit-image) + BFS trace to follow the fiber cables from the node.
4. **Port Detection** — Detects port locations along the traced cable paths.
5. **Callout Generation:**
   - Hardcoded `"NODE"` callout at the detected node location (red border, yellow fill).
   - If `is_connected=True`: Prints hub name + port name callout.
   - If `is_connected=False`: Prints splice can callout.
6. **Survey Info Overlay** — Compact (300pt wide, no page expansion) info block in top-right corner.

---

### 3. Before Map Pipelines (Stateless)

These pipelines **do not run AI models**. They are fast (< 5 seconds) and only stamp information onto maps.

#### Fiber Before Map (`app/services/fiber_before.py`)
- Draws the Survey image in the top-right corner (using `_draw_legend_stack`).
- Draws a yellow/red text box with `PID:`, `NODE:`, `INSTANCE:`, map type, and page count.

#### Coax Before Map (`app/services/coax_before.py`)
- Analyses all 4 corners of the map at 150 DPI to find the **whitest (clearest) corner**.
- If no corner is white enough (map content fills all corners), the page is **automatically extended** upward to create space.
- Stamps the Survey image (700pt wide) + text box in the selected corner.
- Text box uses yellow fill, 1.5pt red border, 14pt black text matching all other map types.

---

## 🤖 AI Models

| Model File | Purpose | Architecture |
|------------|---------|-------------|
| `best.pt` | Main Coax symbol detection | YOLOv8 |
| `power_supply_best.pt` | Power supply detection | YOLOv8 |
| `3x3_4x4_new_model.pt` | Node type classification | YOLOv8 |
| `Internal_best.pt` | Internal splitter detection | YOLOv8 |
| `fiber_node_model.pt` | Fiber node location detection | YOLOv8 |

All models are loaded **once at Celery worker startup** and reused across jobs (singleton pattern) to avoid expensive repeated loading.

---

## 📄 PDF Reporting & Callouts

All callout annotations are **true vector PDF annotations** (not raster overlays), meaning they are:
- Selectable and movable in PDF viewers
- Resolution-independent (look sharp at any zoom)
- Generated using `page.add_freetext_annot()` in PyMuPDF

**Callout styling:**
- **Fill:** Yellow `(1, 1, 0)`
- **Border:** Red `(1, 0, 0)`, `1.5pt` width
- **Text:** Black `(0, 0, 0)`, `10pt` Helvetica
- **Arrow:** Points directly at the detected object

**Deduplication:** Callouts within 50 PDF points (≈ 0.7 inches) with identical text are automatically merged into a single callout.

**Collision Avoidance:** The callout placement engine searches in expanding concentric circles around the target object for empty map space. It avoids:
- Other placed callouts
- The Survey Info block (top-right corner)

---

## 📦 Model Weights

Model weights are **not committed to the repository** (too large for Git). They must be placed manually in the `model_weights/` directory before starting the application.

Contact the project lead for the latest trained model files.

---

*Built with ❤️ by the GP3 internship team.*
