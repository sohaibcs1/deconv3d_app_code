# Deconv3d Web Application

Browser-based AI platform for volumetric microscopy deconvolution.

---

## Setup

### 1. Create Python Environment

```bash
python3 -m venv deconv3d
source deconv3d/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Application

```bash
python3 app.py
```

The application will run on:

```text
http://0.0.0.0:8011
```

---

## Database Management

The application uses SQLite for user sessions and logging.

Open the database:

```bash
sqlite3 users.db
```

List database tables:

```sql
SELECT name FROM sqlite_master WHERE type='table';
```

Check logs table structure:

```sql
PRAGMA table_info(logs);
```

View active users:

```sql
SELECT * FROM users;
```

---

## Session Management

The application uses Flask session-based authentication.

Each user logs in using an email address. The session is signed using the Flask secret key.

Conceptually:

```text
session_data + FLASK_SECRET_KEY → signed_session_cookie
```

For each protected request, the application checks whether the user has a valid session.

If the session is valid:

```text
Request accepted → process function
```

If the session is missing or invalid:

```text
Request rejected → redirect to login
```

---

## Job Storage

Each user receives an isolated job directory:

```text
static/jobs/users/<email_slug>/<job_id>/
```

Each uploaded file is assigned a unique job ID:

```python
job_id = uuid.uuid4().hex[:12]
```

For each job, the application stores:

```text
input file
output file
status.json
meta.json
PNG preview slices
```

---

## Job Retention

Jobs are automatically removed after 10 hours of inactivity.

```python
RETENTION_SEC = 10 * 60 * 60
```

Example:

```text
Current time: 12:00
Last seen time: 10:00
Elapsed time: 2 hours
```

If the elapsed time is greater than 10 hours, the user job folder is deleted and the deletion event is recorded in the logs table.

---

## Multi-User Handling

The application supports multiple users through a job-based backend design.

When a user uploads a volume:

1. The file is saved in the user's job folder.
2. A unique job ID is created.
3. The job status is set to `queued`.
4. A background worker thread starts.
5. GPU inference runs when a slot is available.
6. The job status changes to `done` after successful processing.

GPU concurrency is controlled using:

```python
MAX_CONCURRENT = int(os.getenv("DECONV3D_MAX_CONCURRENT", "2"))
```

This means only two jobs run inference at the same time across all users.

If 100 users submit jobs at the same time:

```text
2 jobs running
98 jobs waiting
```

Waiting jobs are processed in first-come, first-served order as GPU resources become available.

---

## Job Status Flow

Each job moves through the following states:

```text
queued → running → done
```

If an error occurs:

```text
queued → running → error
```

The frontend polls the backend and displays the current job status to the user.

---

## GPU Inference

Inference runs using the configured device:

```python
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

---

## Run and Stop Scripts

Make scripts executable:

```bash
chmod +x run.sh
chmod +x kill.sh
```

Start the application:

```bash
./run.sh
```

Stop the application:

```bash
./kill.sh
```

---

## Notes

The current implementation uses a thread-based first-come, first-served scheduling mechanism. GPU access is controlled using a semaphore, allowing only a fixed number of concurrent inference jobs. Additional jobs remain queued until GPU resources become available.

For large-scale production deployment, the current thread-based job system can be extended using a dedicated queue system such as Celery, Redis Queue, or Kubernetes-based workers.

---

## Citation

<!-- 

```text
Deconv3D: Transformer-guided volumetric deconvolution for high-throughput 3D organoid imaging.
``` -->