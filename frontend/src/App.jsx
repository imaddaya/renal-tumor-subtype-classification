import { useMemo, useRef, useState } from "react";
import "./App.css";

const MODELS = [
  "ResNet18",
  "ResNet18 + MIL",
  "ResNet18 + MIL + Macenko",
  "ResNet18 + MIL + KAN",
  "ResNet18 + Vision Mamba",
  "ResNet18 + Vision Mamba + KAN",
];

const CLASSES = ["chromophobe", "clearcell", "oncocytoma", "papillary"];

function getBackendUrl(modelName) {
  const venvModels = [
    "ResNet18",
    "ResNet18 + MIL",
    "ResNet18 + MIL + Macenko",
    "ResNet18 + MIL + KAN",
  ];

  const newvenvModels = [
    "ResNet18 + Vision Mamba",
    "ResNet18 + Vision Mamba + KAN",
  ];

  if (venvModels.includes(modelName)) return "http://localhost:8001/predict";
  if (newvenvModels.includes(modelName)) return "http://localhost:8002/predict";
  throw new Error(`Unknown model: ${modelName}`);
}

function formatPct(v) {
  return `${(v * 100).toFixed(2)}%`;
}

export default function App() {
  const [selectedModel, setSelectedModel] = useState(MODELS[0]);
  const [trueLabel, setTrueLabel] = useState("chromophobe");
  const [imageFiles, setImageFiles] = useState([]);
  const [savedRows, setSavedRows] = useState([]);
  const [notes, setNotes] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState("");
  const inputRef = useRef(null);

  const previewUrls = useMemo(() => {
    return imageFiles.slice(0, 12).map((file) => ({
      name: file.name,
      url: URL.createObjectURL(file),
    }));
  }, [imageFiles]);

  const probabilities =
    result?.probabilities ?? {
      chromophobe: 0,
      clearcell: 0,
      oncocytoma: 0,
      papillary: 0,
    };

  const predictedLabel = result?.predicted_label ?? "-";
  const confidence = predictedLabel !== "-" ? probabilities[predictedLabel] || 0 : 0;

  const handleFilesChange = (e) => {
    const files = Array.from(e.target.files || []);
    setImageFiles(files);
    setResult(null);
    setErrorMessage("");
  };

  const handlePredict = async () => {
    if (imageFiles.length === 0) {
      setErrorMessage("Please choose patch images first.");
      return;
    }

    if (imageFiles.length < 2) {
      setErrorMessage("Please upload multiple patches from the same WSI, not just one slice.");
      return;
    }

    setIsLoading(true);
    setResult(null);
    setErrorMessage("");

    try {
      const formData = new FormData();
      formData.append("model_name", selectedModel);
      formData.append("true_label", trueLabel);

      imageFiles.forEach((file) => {
        formData.append("images", file);
      });

      const response = await fetch(getBackendUrl(selectedModel), {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error || "Prediction failed.");
      }

      setResult(data);
    } catch (error) {
      setErrorMessage(error.message || "Something went wrong.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSaveResult = () => {
    if (!result || imageFiles.length === 0) return;

    setSavedRows((prev) => [
      {
        id: Date.now().toString(),
        sampleName: imageFiles[0].name,
        patchCount: imageFiles.length,
        model: selectedModel,
        trueLabel,
        predictedLabel: result.predicted_label,
        confidence: formatPct(confidence),
        correct: result.correct,
        chromophobe: formatPct(probabilities.chromophobe),
        clearcell: formatPct(probabilities.clearcell),
        oncocytoma: formatPct(probabilities.oncocytoma),
        papillary: formatPct(probabilities.papillary),
        notes,
      },
      ...prev,
    ]);
  };

  const clearCurrent = () => {
    setImageFiles([]);
    setNotes("");
    setResult(null);
    setErrorMessage("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const removeSaved = (id) => {
    setSavedRows((prev) => prev.filter((row) => row.id !== id));
  };

  return (
    <div className="page">
      <div className="container">
        <h1>RCC WSI Patch Inference Demo</h1>
        <p className="subtitle">
          Upload multiple patches from the same WSI, choose the model and the known correct label, then run inference and compare saved results across models.
        </p>

        <div className="grid">
          <div className="card">
            <h2>Input panel</h2>

            <div className="form-row">
              <div className="field">
                <label>Select model</label>
                <select
                  value={selectedModel}
                  onChange={(e) => {
                    setSelectedModel(e.target.value);
                    setResult(null);
                    setErrorMessage("");
                  }}
                >
                  {MODELS.map((model) => (
                    <option key={model} value={model}>
                      {model}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field">
                <label>Known correct label</label>
                <select
                  value={trueLabel}
                  onChange={(e) => {
                    setTrueLabel(e.target.value);
                    setResult(null);
                    setErrorMessage("");
                  }}
                >
                  {CLASSES.map((label) => (
                    <option key={label} value={label}>
                      {label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            <div className="field">
              <label>Upload patches from one WSI</label>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFilesChange}
              />
              <p className="small-text">
                Upload several patches from the same test WSI. The backend will use up to 70 patches and pad if fewer are provided.
              </p>
            </div>

            <div className="field">
              <label>Notes</label>
              <input
                type="text"
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Optional comment"
              />
            </div>

            <div className="stats-grid single-top-gap">
              <div className="stat-box">
                <span className="stat-label">Selected patches</span>
                <strong>{imageFiles.length}</strong>
              </div>
              <div className="stat-box">
                <span className="stat-label">Target label</span>
                <strong>{trueLabel}</strong>
              </div>
            </div>

            <div className="button-row three">
              <button onClick={handlePredict} disabled={imageFiles.length === 0 || isLoading}>
                {isLoading ? "Running..." : "Run inference"}
              </button>
              <button onClick={handleSaveResult} disabled={!result || isLoading}>
                Save result
              </button>
              <button className="secondary" onClick={clearCurrent}>
                Clear
              </button>
            </div>

            {errorMessage && <div className="error-box">{errorMessage}</div>}
          </div>

          <div className="card">
            <h2>Patch preview</h2>
            {previewUrls.length > 0 ? (
              <>
                <div className="preview-grid">
                  {previewUrls.map((item) => (
                    <div key={item.name} className="thumb-card">
                      <img className="thumb" src={item.url} alt={item.name} />
                      <p className="thumb-name">{item.name}</p>
                    </div>
                  ))}
                </div>
                {imageFiles.length > 12 && (
                  <p className="small-text">Showing first 12 previews out of {imageFiles.length} patches.</p>
                )}
              </>
            ) : (
              <div className="empty-box">No patches selected</div>
            )}
          </div>
        </div>

        <div className="grid">
          <div className="card">
            <h2>Prediction result</h2>

            {!result ? (
              <div className="empty-box">Run inference to see the model prediction.</div>
            ) : (
              <>
                <div className="stats-grid">
                  <div className="stat-box">
                    <span className="stat-label">Known correct label</span>
                    <strong>{trueLabel}</strong>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Predicted label</span>
                    <strong>{predictedLabel}</strong>
                  </div>
                </div>

                <div className="tags">
                  <span className="tag">Model: {selectedModel}</span>
                  <span className="tag">Confidence: {formatPct(confidence)}</span>
                  <span className={result.correct ? "tag success" : "tag danger"}>
                    {result.correct ? "Correct" : "Incorrect"}
                  </span>
                </div>

                <div className="prob-list">
                  {CLASSES.map((label) => (
                    <div key={label} className="prob-item">
                      <div className="prob-header">
                        <span>{label}</span>
                        <span>{formatPct(probabilities[label] || 0)}</span>
                      </div>
                      <div className="bar-bg">
                        <div
                          className="bar-fill"
                          style={{ width: `${(probabilities[label] || 0) * 100}%` }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>

          <div className="card">
            <h2>Saved comparison results</h2>

            {savedRows.length === 0 ? (
              <div className="empty-box">No saved predictions yet.</div>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Sample</th>
                      <th>Patches</th>
                      <th>Model</th>
                      <th>True</th>
                      <th>Predicted</th>
                      <th>Confidence</th>
                      <th>Status</th>
                      <th>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {savedRows.map((row) => (
                      <tr key={row.id}>
                        <td>{row.sampleName}</td>
                        <td>{row.patchCount}</td>
                        <td>{row.model}</td>
                        <td>{row.trueLabel}</td>
                        <td>{row.predictedLabel}</td>
                        <td>{row.confidence}</td>
                        <td>
                          <span className={row.correct ? "tag success" : "tag danger"}>
                            {row.correct ? "Correct" : "Incorrect"}
                          </span>
                        </td>
                        <td>
                          <button className="danger-btn" onClick={() => removeSaved(row.id)}>
                            Remove
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
