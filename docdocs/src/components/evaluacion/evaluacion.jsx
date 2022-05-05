import React, { useState } from "react";
import "./evaluacion.scss";
function Evaluacion() {
  const [predictionMode, setPredictionMode] = useState("fast"); //fast, medium, slow

  return (
    <div className="evaluacion">
      <h2>Evaluar a un paciente</h2>
      <textarea
        placeholder="Clinic history text here..."
        name=""
        id=""
        cols="30"
        rows="10"
      ></textarea>

      <div className="cajas">
        <div
          onClick={() => {
            setPredictionMode("fast");
          }}
          className={`caja ${predictionMode === "fast" ? "selected" : ""} `}
        >
          <div className="icon">
            <ion-icon name="speedometer"></ion-icon>
          </div>
          <div className="text">
            <h3>Predicción Rápida</h3>
            <p className="desc">X pruebas realizadas con este método</p>
            <p className="sub">Naive Bayes</p>
          </div>
        </div>
        <div
          onClick={() => {
            setPredictionMode("medium");
          }}
          className={`caja ${predictionMode === "medium" ? "selected" : ""} `}
        >
          <div className="icon">
            <ion-icon name="sparkles"></ion-icon>
          </div>
          <div className="text">
            <h3>Predicción Balanceada</h3>
            <p className="desc">X pruebas realizadas con este método</p>
            <p className="sub">Logistical Regression</p>
          </div>
        </div>
        <div
          onClick={() => {
            setPredictionMode("slow");
          }}
          className={`caja ${predictionMode === "slow" ? "selected" : ""} `}
        >
          <div className="icon">
            <ion-icon name="pin"></ion-icon>
          </div>
          <div className="text">
            <h3>Predicción Precisa</h3>
            <p className="desc">X pruebas realizadas con este método</p>
            <p className="sub">Support Vector Machines</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Evaluacion;
