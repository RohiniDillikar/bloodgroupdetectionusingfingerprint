import { useState } from "react";
import "./App.css";

function FingerprintVisual() {
  return (
    <div className="visual-stage">

      <div className="glow glow-one"></div>
      <div className="glow glow-two"></div>

      <div className="blood-cell cell-one"></div>
      <div className="blood-cell cell-two"></div>
      <div className="blood-cell cell-three"></div>

      <div className="fingerprint-wrapper">

        <div className="scan-line"></div>

        <svg
          className="fingerprint-svg"
          viewBox="0 0 300 380"
          xmlns="http://www.w3.org/2000/svg"
        >
          <g
            fill="none"
            stroke="currentColor"
            strokeWidth="5"
            strokeLinecap="round"
          >
            <path d="M150 25 C80 25 40 85 40 150 C40 215 65 265 60 330" />
            <path d="M150 45 C95 45 60 90 60 150 C60 210 85 255 78 330" />
            <path d="M150 65 C110 65 82 100 82 150 C82 205 105 240 96 320" />
            <path d="M150 85 C125 85 103 108 103 150 C103 195 125 220 115 305" />
            <path d="M150 105 C135 105 123 120 123 150 C123 185 140 202 132 285" />

            <path d="M150 25 C220 25 260 85 260 150 C260 215 235 265 240 330" />
            <path d="M150 45 C205 45 240 90 240 150 C240 210 215 255 222 330" />
            <path d="M150 65 C190 65 218 100 218 150 C218 205 195 240 204 320" />
            <path d="M150 85 C175 85 197 108 197 150 C197 195 175 220 185 305" />
            <path d="M150 105 C165 105 177 120 177 150 C177 185 160 202 168 285" />

            <path d="M150 125 C130 125 118 138 118 158 C118 180 130 193 150 193 C170 193 182 180 182 158 C182 138 170 125 150 125" />
            <path d="M150 145 C138 145 132 151 132 161 C132 174 140 181 150 181 C160 181 168 174 168 161 C168 151 162 145 150 145" />

            <path d="M115 215 C130 230 170 230 185 215" />
            <path d="M105 235 C125 255 175 255 195 235" />
            <path d="M95 255 C120 280 180 280 205 255" />
          </g>
        </svg>

        <div className="fingerprint-corner corner-tl"></div>
        <div className="fingerprint-corner corner-tr"></div>
        <div className="fingerprint-corner corner-bl"></div>
        <div className="fingerprint-corner corner-br"></div>

      </div>

      <div className="blood-type type-ap">A+</div>
      <div className="blood-type type-op">O+</div>
      <div className="blood-type type-bp">B+</div>
      <div className="blood-type type-ab">AB+</div>
      <div className="blood-type type-am">A−</div>

      <div className="ai-orbit">
        <span>AI</span>
      </div>

    </div>
  );
}


function App() {

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {

    const selectedFile = event.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
  };


  const handlePredict = async () => {

    if (!file) {
      setError("Please upload a fingerprint image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();

    formData.append("file", file);

    try {

      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Prediction failed."
        );
      }

      setResult(data);

    } catch (err) {

      setError(
        "Unable to connect to the prediction server. Make sure FastAPI is running."
      );

    } finally {

      setLoading(false);

    }
  };


  const scrollToPrediction = () => {

    document
      .getElementById("prediction")
      ?.scrollIntoView({
        behavior: "smooth"
      });

  };


  return (

    <div className="app">

      {/* NAVBAR */}

      <header className="navbar">

        <div
          className="brand"
          onClick={() => window.scrollTo({
            top: 0,
            behavior: "smooth"
          })}
        >

          <div className="brand-icon">
            ♥
          </div>

          <span>
            Beyond <strong>Blood Tests</strong>
          </span>

        </div>


        <nav>

          <a href="#home">Home</a>

          <a href="#how-it-works">
            How It Works
          </a>

          <a href="#research">
            Research
          </a>

          <button
            className="nav-button"
            onClick={scrollToPrediction}
          >
            AI Prediction
          </button>

        </nav>

      </header>


      {/* HERO */}

      <main id="home">

        <section className="hero">

          <div className="hero-content">

            <div className="eyebrow">
              DEEP LEARNING&nbsp;&nbsp;×&nbsp;&nbsp;
              COMPUTER VISION&nbsp;&nbsp;×&nbsp;&nbsp;
              BIOMETRICS
            </div>


            <h1>

              BEYOND

              <span>
                BLOOD TESTS
              </span>

            </h1>


            <h2>
              A Deep Learning Approach to{" "}
              <strong>
                Blood Group Identification
              </strong>
              <br />
              via Fingerprint Analysis
            </h2>


            <p className="hero-description">

              Exploring the potential of unique fingerprint
              patterns to identify blood groups using a
              trained ResNet9 deep learning model.

            </p>


            <div className="feature-row">

              <div className="feature">

                <div className="feature-icon">
                  ✦
                </div>

                <div>
                  <strong>
                    AI Powered
                  </strong>

                  <small>
                    ResNet9 Deep Learning
                  </small>
                </div>

              </div>


              <div className="feature">

                <div className="feature-icon">
                  ◉
                </div>

                <div>
                  <strong>
                    Fingerprint Analysis
                  </strong>

                  <small>
                    Pattern Recognition
                  </small>
                </div>

              </div>


              <div className="feature">

                <div className="feature-icon">
                  ♥
                </div>

                <div>
                  <strong>
                    Experimental AI
                  </strong>

                  <small>
                    Blood Group Prediction
                  </small>
                </div>

              </div>

            </div>

          </div>


          <FingerprintVisual />

        </section>


        {/* PREDICTION */}

        <section
          id="prediction"
          className="prediction-section"
        >

          <div className="prediction-card">


            {/* UPLOAD */}

            <div className="upload-panel">

              {!preview ? (

                <div className="upload-box">

                  <div className="upload-symbol">
                    ↑
                  </div>

                  <h3>
                    Upload Fingerprint Image
                  </h3>

                  <p>
                    Drag and drop your fingerprint image
                    here or click to browse
                  </p>

                  <span className="formats">
                    JPG • PNG • BMP • WebP
                  </span>


                  <label className="choose-button">

                    Choose Image

                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      hidden
                    />

                  </label>

                </div>

              ) : (

                <div className="uploaded-state">

                  <img
                    src={preview}
                    alt="Uploaded fingerprint"
                  />

                  <label className="change-button">

                    Change Image

                    <input
                      type="file"
                      accept="image/*"
                      onChange={handleFileChange}
                      hidden
                    />

                  </label>

                </div>

              )}


              <button
                className="predict-button"
                onClick={handlePredict}
                disabled={!file || loading}
              >

                {loading
                  ? "Analyzing Fingerprint..."
                  : "Predict Blood Group"
                }

                {!loading && (
                  <span> →</span>
                )}

              </button>


              {error && (
                <div className="error">
                  {error}
                </div>
              )}

            </div>


            {/* RESULT */}

            <div className="result-panel">

              {result ? (

                <div className="result-content">

                  <div className="result-label">
                    PREDICTION RESULT
                  </div>

                  <div className="result-blood">
                    {result.blood_group}
                  </div>

                  <div className="result-confidence">

                    Confidence:

                    <strong>
                      {" "}
                      {result.confidence}%
                    </strong>

                  </div>


                  <div className="confidence-track">

                    <div
                      className="confidence-progress"
                      style={{
                        width: `${result.confidence}%`
                      }}
                    />

                  </div>


                  <div className="result-message">

                    Model prediction generated
                    successfully.

                  </div>

                </div>

              ) : (

                <div className="empty-result">

                  <div className="result-drop">
                    ♥
                  </div>

                  <span>
                    PREDICTION RESULT
                  </span>

                  <h3>
                    Your result will appear here
                  </h3>

                  <p>
                    Upload a fingerprint image and
                    start the AI analysis.
                  </p>

                </div>

              )}

            </div>

          </div>


          <div className="disclaimer">

            <strong>Important:</strong>{" "}
            This system provides an experimental AI
            prediction and is not a substitute for
            laboratory blood typing or professional
            medical testing.

          </div>

        </section>


        {/* HOW IT WORKS */}

        <section
          id="how-it-works"
          className="info-section"
        >

          <div className="section-heading">

            <span>
              HOW IT WORKS
            </span>

            <h2>
              From Fingerprint to Prediction
            </h2>

          </div>


          <div className="steps">

            <div className="step">

              <div className="step-number">
                01
              </div>

              <h3>
                Upload
              </h3>

              <p>
                Provide a clear fingerprint image
                in a supported format.
              </p>

            </div>


            <div className="step">

              <div className="step-number">
                02
              </div>

              <h3>
                Preprocess
              </h3>

              <p>
                The image is converted and resized
                to the model's input dimensions.
              </p>

            </div>


            <div className="step">

              <div className="step-number">
                03
              </div>

              <h3>
                Deep Learning
              </h3>

              <p>
                The trained ResNet9 model analyzes
                fingerprint patterns.
              </p>

            </div>


            <div className="step">

              <div className="step-number">
                04
              </div>

              <h3>
                Prediction
              </h3>

              <p>
                The model returns a predicted blood
                group and confidence value.
              </p>

            </div>

          </div>

        </section>


        {/* RESEARCH */}

        <section
          id="research"
          className="research-section"
        >

          <div>

            <span className="research-label">
              RESEARCH PROJECT
            </span>

            <h2>
              Exploring AI Beyond
              Conventional Testing
            </h2>

            <p>
              This project investigates whether
              deep learning can identify patterns
              connecting fingerprint characteristics
              with blood group classes.
            </p>

          </div>


          <div className="research-stat">

            <strong>
              8
            </strong>

            <span>
              Blood Group Classes
            </span>

          </div>


          <div className="research-stat">

            <strong>
              128
            </strong>

            <span>
              Pixel Input Size
            </span>

          </div>


          <div className="research-stat">

            <strong>
              ResNet9
            </strong>

            <span>
              Deep Learning Model
            </span>

          </div>

        </section>

      </main>


      <footer>

        <span>
          Beyond Blood Tests
        </span>

        <p>
          Deep Learning Research Project • Fingerprint Analysis
        </p>

      </footer>

    </div>
  );
}

export default App;