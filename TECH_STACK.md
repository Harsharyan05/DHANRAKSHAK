# 🛠 Tech Stack — Dhanrakshak ERP Lite

This document details the technologies, libraries, and tools used to build **Dhanrakshak ERP Lite**.

---

## 1. Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      TECH STACK                             │
│                                                             │
│   Frontend      │  React.js · HTML5 · CSS3 · Chart.js      │
│   Backend       │  Node.js (Express) / Python (FastAPI)     │
│   Database      │  PostgreSQL · MySQL                       │
│   Auth          │  JWT · bcrypt                             │
│   Cloud         │  AWS · Azure · GCP                        │
│   Deployment    │  Vercel · Docker · CI/CD                  │
│   Dev Tools     │  Git · ESLint · Prettier · VS Code        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Frontend

| Technology | Version | Purpose |
|---|---|---|
| ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) **React.js** | 18.x+ | Component-based SPA framework for building the dashboard UI |
| ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) **HTML5** | — | Semantic markup and page structure |
| ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) **CSS3** | — | Responsive styling, flexbox/grid layouts, animations |
| ![JS](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black) **JavaScript (ES6+)** | ES2022+ | Application logic, async operations, DOM interactions |
| **React Router** | 6.x+ | Client-side routing and navigation |
| **Recharts / Chart.js** | Latest | Interactive charts, graphs, and KPI visualizations |
| **Axios** | 1.x+ | HTTP client for API communication |

### Why React.js?

- **Component-based architecture** — Reusable UI components across modules
- **Virtual DOM** — Fast rendering for data-heavy dashboards
- **Rich ecosystem** — Extensive library support for charts, forms, and state management
- **Large community** — Easy to find developers and resources for MSMEs

---

## 3. Backend

| Technology | Version | Purpose |
|---|---|---|
| ![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white) **Node.js** | 18.x+ | Server-side JavaScript runtime |
| **Express.js** | 4.x+ | Lightweight REST API framework |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) **Python** *(alternative)* | 3.10+ | Alternative backend runtime |
| **FastAPI** *(alternative)* | 0.100+ | High-performance Python API framework |
| **Django** *(alternative)* | 4.x+ | Full-featured Python web framework |

### Why Node.js / Python?

| Criteria | Node.js (Express) | Python (FastAPI) |
|---|---|---|
| **Performance** | Non-blocking I/O, great for real-time | Async support, excellent for computation |
| **Learning Curve** | Easy for frontend devs (same language) | Clean syntax, beginner-friendly |
| **Ecosystem** | npm — largest package registry | pip — strong data/ML library support |
| **Best For** | Real-time dashboards, API-heavy apps | Data processing, AI/ML integration |

---

## 4. Database

| Technology | Version | Purpose |
|---|---|---|
| ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white) **PostgreSQL** | 15.x+ | Primary relational database (recommended) |
| ![MySQL](https://img.shields.io/badge/MySQL-005C84?style=flat&logo=mysql&logoColor=white) **MySQL** | 8.x+ | Alternative relational database |
| **Redis** *(future)* | 7.x+ | In-memory caching for dashboards and sessions |

### Why PostgreSQL?

- **ACID compliance** — Reliable transactions for financial and inventory data
- **Advanced features** — JSON support, full-text search, window functions
- **Scalability** — Read replicas, partitioning, connection pooling
- **Open-source** — No licensing costs, perfect for MSME budgets

### ORM Layer

| Stack | ORM | Purpose |
|---|---|---|
| Node.js | **Sequelize** / **Prisma** | Object-relational mapping, migrations, query building |
| Python | **SQLAlchemy** / **Django ORM** | Database abstraction, schema management |

---

## 5. Authentication & Security

| Technology | Purpose |
|---|---|
| ![JWT](https://img.shields.io/badge/JWT-000000?style=flat&logo=jsonwebtokens&logoColor=white) **JSON Web Tokens (JWT)** | Stateless API authentication |
| **bcrypt / argon2** | Secure password hashing |
| **Helmet.js** *(Node.js)* | HTTP security headers |
| **CORS middleware** | Cross-origin request control |
| **express-rate-limit** | API rate limiting to prevent abuse |
| **HTTPS / TLS** | Encrypted data in transit |

### Auth Flow

```
User Login → Server validates credentials → Server issues JWT (access + refresh)
    → Client stores tokens → Client sends JWT with each API request
    → Server verifies JWT → Grants/denies access based on role
```

---

## 6. DevOps & Tooling

| Tool | Purpose |
|---|---|
| ![Git](https://img.shields.io/badge/Git-F05032?style=flat&logo=git&logoColor=white) **Git** | Version control |
| ![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white) **GitHub** | Code hosting, issues, PRs, CI/CD |
| **ESLint** | JavaScript linting and code quality |
| **Prettier** | Consistent code formatting |
| ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?style=flat&logo=visualstudiocode&logoColor=white) **VS Code** | Recommended IDE |
| **Postman** | API testing and documentation |
| **npm** | Package management |

---

## 7. Testing

| Tool | Type | Purpose |
|---|---|---|
| **Jest** | Unit Testing | Test individual functions and components |
| **React Testing Library** | Component Testing | Test React component behavior |
| **Supertest** | API Testing | Test Express API endpoints |
| **Cypress** *(optional)* | E2E Testing | Full end-to-end browser testing |
| **pytest** *(Python)* | Unit Testing | Test Python backend logic |

---

## 8. Future Technology Additions

| Technology | Purpose | Timeline |
|---|---|---|
| **React Native** | Mobile apps (iOS & Android) | Phase 2 |
| **Redis** | Caching layer for performance | Phase 2 |
| **RabbitMQ / Kafka** | Event-driven message queues | Phase 3 |
| **TensorFlow / scikit-learn** | AI/ML models for forecasting | Phase 3 |
| **MQTT** | IoT device communication | Phase 3 |
| **GraphQL** | Efficient mobile data fetching | Phase 3 |
| **Docker** | Containerized deployments | Phase 2 |
| **Kubernetes** | Container orchestration at scale | Phase 3 |

---

## 9. Stack Compatibility Matrix

| Component | Minimum | Recommended | Notes |
|---|---|---|---|
| Node.js | 16.x | 18.x LTS+ | Use even-numbered LTS releases |
| npm | 8.x | 9.x+ | Comes with Node.js |
| React | 17.x | 18.x+ | For concurrent features |
| PostgreSQL | 13.x | 15.x+ | Best performance and features |
| MySQL | 5.7 | 8.x+ | JSON support requires 5.7+ |
| Python | 3.8 | 3.10+ | If using FastAPI/Django backend |
| Browser | ES6 support | Chrome/Edge/Firefox latest | IE not supported |

---

<p align="center">
  <em>Stack chosen for simplicity, affordability, and scalability — perfect for MSMEs.</em>
</p>
