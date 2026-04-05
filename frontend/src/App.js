import React, { useState, useEffect, useCallback } from "react";

const API_BASE = process.env.REACT_APP_API_URL || "http://localhost:8000";

const styles = {
  app: { minHeight: "100vh", background: "#1a1a2e", color: "#eee", padding: "24px" },
  header: { textAlign: "center", marginBottom: "32px" },
  title: { fontSize: "2rem", fontWeight: "bold", color: "#e94560" },
  subtitle: { color: "#aaa", marginTop: "8px" },
  card: {
    background: "#16213e", borderRadius: "12px", padding: "24px",
    marginBottom: "20px", boxShadow: "0 4px 20px rgba(0,0,0,0.3)",
  },
  cardTitle: { fontSize: "1.1rem", fontWeight: "600", marginBottom: "16px", color: "#e94560" },
  grid: { display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(300px, 1fr))", gap: "20px" },
  input: {
    width: "100%", padding: "10px 14px", borderRadius: "8px",
    border: "1px solid #333", background: "#0f3460", color: "#eee",
    fontSize: "14px", marginBottom: "12px",
  },
  button: {
    background: "#e94560", color: "#fff", border: "none", borderRadius: "8px",
    padding: "10px 24px", cursor: "pointer", fontWeight: "600", width: "100%",
  },
  badge: (dir) => ({
    display: "inline-block", padding: "4px 12px", borderRadius: "20px", fontWeight: "bold",
    background: dir === "UP" ? "#0a7c4c" : dir === "DOWN" ? "#7c0a2a" : "#3a3a5c",
    color: "#fff", fontSize: "0.85rem",
  }),
  stat: { display: "flex", justifyContent: "space-between", padding: "8px 0", borderBottom: "1px solid #2a2a4a" },
  statusDot: (ok) => ({
    display: "inline-block", width: "10px", height: "10px", borderRadius: "50%",
    background: ok ? "#2ecc71" : "#e74c3c", marginRight: "8px",
  }),
};

function HealthCard() {
  const [health, setHealth] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then(setHealth)
      .catch(() => setHealth({ status: "unreachable" }))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div style={styles.card}>
      <div style={styles.cardTitle}>API Status</div>
      {loading ? (
        <p style={{ color: "#aaa" }}>Checking…</p>
      ) : (
        <>
          <div style={styles.stat}>
            <span>Status</span>
            <span>
              <span style={styles.statusDot(health.status === "healthy")} />
              {health.status}
            </span>
          </div>
          <div style={styles.stat}>
            <span>Database</span>
            <span>{health.db || "—"}</span>
          </div>
          <div style={{ ...styles.stat, borderBottom: "none" }}>
            <span>WS Connections</span>
            <span>{health.websocket_connections ?? "—"}</span>
          </div>
        </>
      )}
    </div>
  );
}

function PredictCard() {
  const [symbol, setSymbol] = useState("AAPL");
  const [entryPrice, setEntryPrice] = useState("180");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handlePredict = useCallback(async () => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const resp = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol, entry_price: parseFloat(entryPrice) }),
      });
      const data = await resp.json();
      if (!resp.ok) throw new Error(data.detail || "Request failed");
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [symbol, entryPrice]);

  return (
    <div style={styles.card}>
      <div style={styles.cardTitle}>Price Prediction</div>
      <input
        style={styles.input}
        value={symbol}
        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
        placeholder="Symbol (e.g. AAPL)"
      />
      <input
        style={styles.input}
        type="number"
        value={entryPrice}
        onChange={(e) => setEntryPrice(e.target.value)}
        placeholder="Entry Price"
      />
      <button style={styles.button} onClick={handlePredict} disabled={loading}>
        {loading ? "Predicting…" : "Predict"}
      </button>
      {error && <p style={{ color: "#e94560", marginTop: "12px" }}>{error}</p>}
      {result && (
        <div style={{ marginTop: "16px" }}>
          <div style={styles.stat}>
            <span>Direction</span>
            <span style={styles.badge(result.direction)}>{result.direction}</span>
          </div>
          <div style={styles.stat}>
            <span>Confidence</span>
            <span>{result.confidence != null ? `${(result.confidence * 100).toFixed(1)}%` : "—"}</span>
          </div>
          <div style={styles.stat}>
            <span>Stop Loss</span>
            <span>{result.stop_loss != null ? `$${result.stop_loss.toFixed(2)}` : "—"}</span>
          </div>
          <div style={{ ...styles.stat, borderBottom: "none" }}>
            <span>Take Profit</span>
            <span>{result.take_profit != null ? `$${result.take_profit.toFixed(2)}` : "—"}</span>
          </div>
        </div>
      )}
    </div>
  );
}

export default function App() {
  return (
    <div style={styles.app}>
      <div style={styles.header}>
        <h1 style={styles.title}>📈 Trading Predictor</h1>
        <p style={styles.subtitle}>ML-powered trading signals with risk management</p>
      </div>
      <div style={styles.grid}>
        <HealthCard />
        <PredictCard />
      </div>
    </div>
  );
}
