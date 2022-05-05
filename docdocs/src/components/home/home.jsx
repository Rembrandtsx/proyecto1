import React from "react";
import "./home.scss";
function Home() {
  return (
    <div className="home">
      <div className="row">
        <div className="saludo">
          <h2>Hola Camilo</h2>
          <p>Aquí podrás hacer blah blah blah</p>
        </div>
        <div className="numbers up">
          <h4>105</h4>
          <p>Candidatos Aceptados</p>
        </div>
        <div className="numbers down">
          <h4>20</h4>
          <p>Candidatos Rechazados</p>
        </div>
      </div>
      <div className="row2">
        <h2>Categorías</h2>
        <div className="cajas">
          <div className="caja">
            <div className="icon">
              <ion-icon name="speedometer"></ion-icon>
            </div>
            <div className="text">
              <h3>Predicción Rápida</h3>
              <p className="desc">X pruebas realizadas con este método</p>
              <p className="sub">Naive Bayes</p>
            </div>
          </div>
          <div className="caja">
            <div className="icon">
              <ion-icon name="sparkles"></ion-icon>
            </div>
            <div className="text">
              <h3>Predicción Balanceada</h3>
              <p className="desc">X pruebas realizadas con este método</p>
              <p className="sub">Logistical Regression</p>
            </div>
          </div>
          <div className="caja">
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
      <div className="bottom">
        <h2>Research in progress</h2>
      </div>
    </div>
  );
}

export default Home;
