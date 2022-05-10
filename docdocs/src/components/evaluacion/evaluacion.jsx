import React, { useEffect, useState } from "react";
import "./evaluacion.scss";
import userData from "../../assets/UserData.json";
import users from "../../assets/Users.json";

function Evaluacion() {
  const [predictionMode, setPredictionMode] = useState("fast"); //fast, medium, slow
  const [filingType, setFilingType] = useState("menu"); //menu, empty, filled
  const [label, setLabel] = useState("__label__0");
  const [text, setText] = useState("");
  const [disabled, setDisabled] = useState(false);
  const [userInfo, setuserInfo] = useState([]);
  useEffect(() => {
    setuserInfo(
      userData.map((el, i) => {
        return { ...el, ...users[i] };
      })
    );

    return () => {};
  }, []);

  const changeLabel = (e) => {
    setLabel(e.target.value);
  };

  const changeText = (e) => {
    setText(e.target.value);
  };

  const setInfoIndex = (i) => {
    setLabel(userInfo[i].label);
    setText(userInfo[i].study_and_condition);
    setFilingType("empty");
    setDisabled(true);
  };

  const analizar = () => {};

  if (filingType === "menu") {
    return (
      <div className="evaluacion">
        <h2>Evaluar a un paciente</h2>
        <div className="cards">
          <div
            className="card"
            onClick={() => {
              setFilingType("filled");
            }}
          >
            <ion-icon name="list-outline"></ion-icon>
            <p>Elegir de los pacientes registrados</p>
          </div>
          <div
            className="card"
            onClick={() => {
              setFilingType("empty");
            }}
          >
            <ion-icon name="add-outline"></ion-icon>
            <p>Llenar un nuevo registro</p>
          </div>
        </div>
      </div>
    );
  } else if (filingType === "filled") {
    return (
      <div className="evaluacion">
        <div className="messages-section">
          <div className="projects-section-header">
            <h2>Evaluar a un paciente</h2>
          </div>
          <div className="messages">
            {userInfo.map((el, i) => {
              if (i < 200)
                return (
                  <div
                    key={i}
                    onClick={() => {
                      setInfoIndex(i);
                    }}
                    className="message-box"
                  >
                    <img src={el.img} alt="profile" />

                    <div className="message-content">
                      <div className="message-header">
                        <div className="name">{el.name}</div>
                      </div>
                      <p className="message-line">
                        <b>{el.label} :</b>
                        {el.study_and_condition}
                      </p>
                      <p className="message-line time">{el.gender}</p>
                    </div>
                  </div>
                );
              else return null;
            })}
          </div>
        </div>

        <div className="list"></div>
      </div>
    );
  } else {
    return (
      <div className="evaluacion">
        <h2>Evaluar a un paciente</h2>
        <h4>Label del estudio</h4>
        <div className="select">
          <select
            value={label}
            onChange={changeLabel}
            disabled={disabled}
            name=""
            id=""
          >
            <option value="__label__0">__label__0</option>
            <option value="__label__1">__label__1</option>
          </select>
        </div>
        <h4>Reporte clínico</h4>
        <textarea
          value={text}
          onChange={changeText}
          disabled={disabled}
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
        <div className="btn-container">
          <button onClick={analizar} className="btn main-btn">
            Evaluar Elegibilidad del Paciente
          </button>
        </div>
      </div>
    );
  }
}

export default Evaluacion;
