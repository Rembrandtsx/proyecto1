import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/sidebar/sidebar";
import "./App.scss";
import Home from "./components/home/home";
import Login from "./components/login/login";
import Evaluacion from "./components/evaluacion/evaluacion";

function App() {
  return (
    <div className="contenedor">
      <Sidebar></Sidebar>
      <main>
        <Routes>
          <Route index element={<Home />} />
          <Route path="login" element={<Login />} />
          <Route path="evaluar" element={<Evaluacion />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
