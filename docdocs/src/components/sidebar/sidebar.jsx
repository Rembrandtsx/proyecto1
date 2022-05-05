import React from "react";
import "./sidebar.scss";
import logo from "../../assets/DOCDOCS.png";
import { Link, NavLink } from "react-router-dom";

function Sidebar() {
  return (
    <aside>
      <div className="nav">
        <div className="logo">
          <img src={logo} alt="" />
        </div>
        <ul>
          <NavLink
            className={({ isActive }) => (isActive ? "active" : "")}
            to="/home"
          >
            <li>
              <ion-icon name="grid"></ion-icon>
              <p>Dashboard</p>
            </li>
          </NavLink>
          <NavLink
            className={({ isActive }) => (isActive ? "active" : "")}
            to="/evaluar"
          >
            <li>
              <ion-icon name="medkit"></ion-icon>
              <p>Evaluar</p>
            </li>
          </NavLink>
          <NavLink
            className={({ isActive }) => (isActive ? "active" : "")}
            to="/login"
          >
            <li>
              <ion-icon name="person"></ion-icon>
              <p>Perfil</p>
            </li>
          </NavLink>

          <NavLink className="last" to="/home">
            <li>
              <ion-icon name="clipboard"></ion-icon>
              <p>Evaluar un paciente</p>
            </li>
          </NavLink>
        </ul>
      </div>
    </aside>
  );
}

export default Sidebar;
