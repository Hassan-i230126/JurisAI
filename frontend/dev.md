# Frontend Development Notes

For hot-reload development, run both the Vite dev server and FastAPI in parallel.

## 1) Frontend

```bash
cd frontend
npm run dev
```

Vite runs on port `5173`.

## 2) Backend

```bash
cd backend
uvicorn app.main:app --reload
```

Vite proxies `/ws` and `/api` requests to `localhost:8000` via `vite.config.js`.
