import { useEffect, useMemo, useRef, useState } from "react";
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

const MIN_PATCHES = 70;
const MAX_PATCHES = 500;

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
  return `${(Number(v || 0) * 100).toFixed(2)}%`;
}

function formatBackendError(data) {
  if (!data) return "Prediction failed.";

  if (typeof data.error === "string") return data.error;
  if (typeof data.detail === "string") return data.detail;

  if (data.detail && typeof data.detail === "object") {
    if (typeof data.detail.error === "string") return data.detail.error;

    if (
      data.detail.required_minimum_usable &&
      data.detail.usable_count !== undefined
    ) {
      return `Not enough usable patches after filtering. Usable: ${data.detail.usable_count}, required: ${data.detail.required_minimum_usable}.`;
    }

    return JSON.stringify(data.detail, null, 2);
  }

  if (typeof data.message === "string") return data.message;

  return "Prediction failed.";
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

  useEffect(() => {
    return () => {
      previewUrls.forEach((item) => URL.revokeObjectURL(item.url));
    };
  }, [previewUrls]);

  const probabilities = result?.probabilities ?? {
    chromophobe: 0,
    clearcell: 0,
    oncocytoma: 0,
    papillary: 0,
  };

  const predictedLabel = result?.predicted_label ?? "-";
  const confidence =
    predictedLabel !== "-" ? probabilities[predictedLabel] || 0 : 0;

  const usedPatches = result?.used_patches ?? [];
  const topImportantPatches = result?.top_important_patches ?? [];
  const selectionSummary = result?.selection_summary ?? null;

  const handleFilesChange = (e) => {
    const files = Array.from(e.target.files || []);
    setImageFiles(files);
    setResult(null);
    setErrorMessage("");
  };

  const handlePredict = async () => {
    if (imageFiles.length < MIN_PATCHES || imageFiles.length > MAX_PATCHES) {
      setErrorMessage(
        `Please upload between ${MIN_PATCHES} and ${MAX_PATCHES} patches from the same biopsy/WSI.`
      );
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

      if (!response.ok) {
        throw new Error(formatBackendError(data));
      }

      if (data?.error) {
        throw new Error(formatBackendError(data));
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
        sampleName: imageFiles[0]?.name || "sample",
        patchCountUploaded: result.patch_count_uploaded ?? imageFiles.length,
        patchCountUsed: result.patch_count_used ?? 0,
        patchCountDropped: result.patch_count_dropped ?? 0,
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
          Upload patches from one biopsy / one WSI, choose a model and the known
          label, then run inference and compare results across models.
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
              <label>Upload patches from one WSI / one biopsy</label>
              <input
                ref={inputRef}
                type="file"
                accept="image/*"
                multiple
                onChange={handleFilesChange}
              />
              <p className="small-text">
                Upload between {MIN_PATCHES} and {MAX_PATCHES} patches. The
                backend will filter low-tissue / very white patches, sort by
                tissue content, and use the best 70 patches for inference.
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
                <span className="stat-label">Uploaded patches</span>
                <strong>{imageFiles.length}</strong>
              </div>
              <div className="stat-box">
                <span className="stat-label">Allowed range</span>
                <strong>
                  {MIN_PATCHES} - {MAX_PATCHES}
                </strong>
              </div>
            </div>

            <div className="button-row three">
              <button
                onClick={handlePredict}
                disabled={
                  isLoading ||
                  imageFiles.length < MIN_PATCHES ||
                  imageFiles.length > MAX_PATCHES
                }
              >
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
                  <p className="small-text">
                    Showing first 12 previews out of {imageFiles.length} patches.
                  </p>
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
              <div className="empty-box">
                Run inference to see the model prediction.
              </div>
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
                  <div className="stat-box">
                    <span className="stat-label">Uploaded</span>
                    <strong>{result.patch_count_uploaded}</strong>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Used</span>
                    <strong>{result.patch_count_used}</strong>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Dropped / Not used</span>
                    <strong>{result.patch_count_dropped}</strong>
                  </div>
                  <div className="stat-box">
                    <span className="stat-label">Confidence</span>
                    <strong>{formatPct(confidence)}</strong>
                  </div>
                </div>

                <div className="tags">
                  <span className="tag">Model: {selectedModel}</span>
                  <span className="tag">
                    Patch importance:{" "}
                    {result.patch_importance_supported ? "Supported" : "Not supported"}
                  </span>
                  <span className={result.correct ? "tag success" : "tag danger"}>
                    {result.correct ? "Correct" : "Incorrect"}
                  </span>
                </div>

                {selectionSummary && (
                  <>
                    <h3>Selection summary</h3>
                    <div className="table-wrap">
                      <table>
                        <tbody>
                          <tr>
                            <th>Uploaded count</th>
                            <td>{selectionSummary.uploaded_count}</td>
                          </tr>
                          <tr>
                            <th>Usable after filtering</th>
                            <td>{selectionSummary.usable_count_after_filtering}</td>
                          </tr>
                          <tr>
                            <th>Dropped count</th>
                            <td>{selectionSummary.dropped_count}</td>
                          </tr>
                          <tr>
                            <th>Target patch count</th>
                            <td>{selectionSummary.target_patch_count}</td>
                          </tr>
                          <tr>
                            <th>Selection rule</th>
                            <td>{selectionSummary.selection_rule}</td>
                          </tr>
                        </tbody>
                      </table>
                    </div>
                  </>
                )}

                <h3>Class probabilities</h3>
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

                <h3>Used patches</h3>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Rank</th>
                        <th>Filename</th>
                        <th>White %</th>
                        <th>Tissue %</th>
                      </tr>
                    </thead>
                    <tbody>
                      {usedPatches.map((patch) => (
                        <tr key={`${patch.rank}-${patch.filename}`}>
                          <td>{patch.rank}</td>
                          <td>{patch.filename}</td>
                          <td>{patch.white_percent}</td>
                          <td>{patch.tissue_percent}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {result.patch_importance_supported && topImportantPatches?.length > 0 && (
                  <>
                    <h3>Top important patches</h3>
                    <div className="table-wrap">
                      <table>
                        <thead>
                          <tr>
                            <th>Filename</th>
                            <th>Attention weight</th>
                            <th>White %</th>
                            <th>Tissue %</th>
                          </tr>
                        </thead>
                        <tbody>
                          {topImportantPatches.map((patch) => (
                            <tr key={`${patch.filename}-${patch.attention_weight}`}>
                              <td>{patch.filename}</td>
                              <td>{patch.attention_weight}</td>
                              <td>{patch.white_percent}</td>
                              <td>{patch.tissue_percent}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </>
                )}
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
                      <th>Uploaded</th>
                      <th>Used</th>
                      <th>Dropped</th>
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
                        <td>{row.patchCountUploaded}</td>
                        <td>{row.patchCountUsed}</td>
                        <td>{row.patchCountDropped}</td>
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
                          <button
                            className="danger-btn"
                            onClick={() => removeSaved(row.id)}
                          >
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