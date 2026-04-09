# 🏗 System Architecture — Dhanrakshak ERP Lite

This document describes the system architecture of **Dhanrakshak ERP Lite**, an affordable ERP framework designed for MSME manufacturing enterprises.

---

## 1. Architecture Overview

Dhanrakshak ERP Lite follows a **modular, layered architecture** that cleanly separates concerns across presentation, business logic, data access, and analytics layers. Each ERP module (Inventory, Production, Procurement, Sales, Finance) is an independent service that communicates through a unified RESTful API layer.

```
┌─────────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                           │
│                                                                 │
│   ┌─────────────┐   ┌──────────────┐   ┌───────────────────┐   │
│   │  Web App    │   │ Mobile App   │   │  Admin Dashboard  │   │
│   │  (React.js) │   │  (Future)    │   │  (Role-Based)     │   │
│   └──────┬──────┘   └──────┬───────┘   └────────┬──────────┘   │
│          └─────────────────┼────────────────────┘               │
└────────────────────────────┼────────────────────────────────────┘
                             │  HTTPS / REST
┌────────────────────────────▼────────────────────────────────────┐
│                     API GATEWAY LAYER                           │
│                                                                 │
│          ┌──────────────────────────────────┐                   │
│          │  Authentication & Authorization  │                   │
│          │         (JWT Tokens)             │                   │
│          └──────────────┬───────────────────┘                   │
│                         │                                       │
│          ┌──────────────▼───────────────────┐                   │
│          │     Request Routing & Validation │                   │
│          └──────────────┬───────────────────┘                   │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                  APPLICATION / SERVICE LAYER                    │
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐              │
│  │  Inventory  │ │ Production  │ │ Procurement  │              │
│  │   Service   │ │   Service   │ │   Service    │              │
│  └─────────────┘ └─────────────┘ └──────────────┘              │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐              │
│  │   Sales     │ │  Finance    │ │    User      │              │
│  │   Service   │ │   Service   │ │   Service    │              │
│  └─────────────┘ └─────────────┘ └──────────────┘              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                    DATA ACCESS LAYER                            │
│                                                                 │
│          ┌──────────────────────────────────┐                   │
│          │        ORM / Query Builder       │                   │
│          └──────────────┬───────────────────┘                   │
└─────────────────────────┼───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│                     DATABASE LAYER                              │
│                                                                 │
│   ┌──────────────────┐       ┌──────────────────────────┐       │
│   │   PostgreSQL /   │       │   Cache Layer (Future)   │       │
│   │      MySQL       │       │      (Redis)             │       │
│   └──────────────────┘       └──────────────────────────┘       │
└─────────────────────────┬───────────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────────┐
│               ANALYTICS & REPORTING ENGINE                      │
│                                                                 │
│   ┌──────────────┐  ┌───────────────┐  ┌────────────────────┐   │
│   │  Dashboards  │  │  Report Gen   │  │  AI/ML Engine      │   │
│   │  & Charts    │  │  (PDF/Excel)  │  │  (Future Phase)    │   │
│   └──────────────┘  └───────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Layer Descriptions

### 2.1 Presentation Layer

The user-facing layer that provides interactive dashboards and forms.

| Component | Technology | Description |
|---|---|---|
| Web Application | React.js | Primary SPA for all ERP operations |
| Admin Dashboard | React.js | Role-based admin panel for system configuration |
| Mobile App | React Native *(future)* | Mobile-friendly interface for on-the-go access |

**Key Responsibilities:**
- Render dynamic UI components (tables, charts, forms)
- Handle client-side form validation
- Manage application state and routing
- Responsive design for desktop and mobile browsers

---

### 2.2 API Gateway Layer

Central entry point for all client requests, enforcing security and routing.

| Component | Description |
|---|---|
| **JWT Authentication** | Verifies user identity via signed tokens |
| **Role-Based Authorization** | Enforces access control (Admin, Manager, Supervisor, etc.) |
| **Request Validation** | Validates incoming payloads before forwarding to services |
| **Rate Limiting** | Prevents abuse and ensures fair usage |

---

### 2.3 Application / Service Layer

The business logic layer, organized into **independent, modular services**:

| Service | Responsibilities |
|---|---|
| **Inventory Service** | Stock CRUD, warehouse tracking, reorder alerts, stock reports |
| **Production Service** | Work order lifecycle, machine scheduling, production tracking |
| **Procurement Service** | Supplier management, purchase orders, delivery tracking |
| **Sales Service** | Customer orders, invoicing, delivery status, sales analytics |
| **Finance Service** | Expense/revenue tracking, profit margins, cash flow reports |
| **User Service** | Registration, login, profile management, role assignment |

Each service is **loosely coupled** and communicates through well-defined internal interfaces, allowing independent development and scaling.

---

### 2.4 Data Access Layer

Abstracts database interactions from business logic.

- **ORM / Query Builder** — Maps application objects to database tables
- **Migration System** — Version-controlled schema changes
- **Connection Pooling** — Efficient database connection management

---

### 2.5 Database Layer

| Store | Purpose |
|---|---|
| **PostgreSQL / MySQL** | Primary relational data store for all transactional data |
| **Redis** *(future)* | Caching layer for frequently accessed data (dashboards, reports) |

---

### 2.6 Analytics & Reporting Engine

| Component | Description |
|---|---|
| **Dashboards & Charts** | Visual KPIs for inventory levels, production status, financials |
| **Report Generation** | Export reports in PDF / Excel format |
| **AI/ML Engine** *(future)* | Demand forecasting, inventory prediction, production optimization |

---

## 3. Data Flow Diagram

```
   ┌────────┐         ┌───────────┐         ┌──────────────┐
   │  User  │──────►  │  React UI │──────►  │  API Gateway │
   └────────┘         └───────────┘         └──────┬───────┘
                                                   │
                                          ┌────────▼────────┐
                                          │  JWT Validation  │
                                          └────────┬────────┘
                                                   │
                            ┌──────────────────────┼──────────────────────┐
                            │                      │                      │
                   ┌────────▼──────┐     ┌────────▼──────┐     ┌────────▼──────┐
                   │   Inventory   │     │  Production   │     │    Sales      │
                   │    Service    │     │   Service     │     │   Service     │
                   └────────┬──────┘     └────────┬──────┘     └────────┬──────┘
                            │                      │                      │
                            └──────────────────────┼──────────────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │    Database     │
                                          │  (PostgreSQL)   │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   Analytics &   │
                                          │   Reporting     │
                                          └─────────────────┘
```

---

## 4. Module Interaction Map

The ERP modules interact with each other to support end-to-end manufacturing workflows:

```
                        ┌──────────────────┐
                        │  User Management │
                        │  (Auth & Roles)  │
                        └────────┬─────────┘
                                 │ secures all modules
         ┌───────────────────────┼───────────────────────┐
         │                       │                       │
┌────────▼──────┐      ┌────────▼──────┐      ┌────────▼──────┐
│   Inventory   │◄────►│  Production   │      │  Procurement  │
│  Management   │      │  Management   │      │  Management   │
└────────┬──────┘      └────────┬──────┘      └────────┬──────┘
         │                      │                       │
         │  stock consumed      │  costs recorded       │  purchase costs
         │  by production       │  per work order       │  tracked
         │                      │                       │
         └──────────┬───────────┴───────────┬───────────┘
                    │                       │
           ┌────────▼──────┐      ┌────────▼──────┐
           │    Sales &    │      │   Finance     │
           │    Orders     │─────►│   Dashboard   │
           └───────────────┘      └───────────────┘
                 revenue ────────►  revenue + expenses
                 recorded           = profit analysis
```

**Key interactions:**
- **Inventory ↔ Production** — Raw materials are consumed during production; finished goods are added back to inventory.
- **Procurement → Inventory** — Purchased materials increase inventory stock levels.
- **Production → Finance** — Production costs (labor, materials, machine time) feed into expense tracking.
- **Sales → Finance** — Revenue from fulfilled orders flows into the financial dashboard.
- **Procurement → Finance** — Purchase costs are recorded as expenses.

---

## 5. Authentication & Authorization Flow

```
  ┌────────┐     1. Login Request        ┌────────────┐
  │  User  │ ──────────────────────────► │  Auth API  │
  └────────┘                             └─────┬──────┘
       ▲                                       │
       │      2. Validate Credentials          │
       │         (DB Lookup)                   ▼
       │                                ┌─────────────┐
       │      3. Return JWT Token       │  Database   │
       │ ◄───────────────────────────── └─────────────┘
       │
       │      4. Attach JWT to
       │         subsequent requests
       ▼
  ┌──────────┐    5. Verify Token     ┌──────────────┐
  │ React UI │ ─────────────────────► │ API Gateway  │
  └──────────┘                        └──────┬───────┘
                                             │
                  6. Check Role              │
                     Permissions             ▼
                                      ┌──────────────┐
                                      │   Service    │
                                      │   Layer      │
                                      └──────────────┘
```

### Role-Based Access Matrix

| Feature | Admin | Manager | Supervisor | Viewer |
|---|:---:|:---:|:---:|:---:|
| User Management | ✅ | ❌ | ❌ | ❌ |
| Inventory — Full CRUD | ✅ | ✅ | ✅ | ❌ |
| Inventory — View Only | ✅ | ✅ | ✅ | ✅ |
| Production — Manage Work Orders | ✅ | ✅ | ✅ | ❌ |
| Procurement — Manage Suppliers | ✅ | ✅ | ❌ | ❌ |
| Sales — Create Orders | ✅ | ✅ | ❌ | ❌ |
| Finance Dashboard | ✅ | ✅ | ❌ | ❌ |
| Reports & Analytics | ✅ | ✅ | ✅ | ✅ |
| System Settings | ✅ | ❌ | ❌ | ❌ |

---

## 6. Deployment Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Cloud Provider                         │
│              (AWS / Azure / GCP)                         │
│                                                          │
│   ┌─────────────┐    ┌──────────────┐   ┌────────────┐  │
│   │   CDN       │    │  Web Server  │   │  Database  │  │
│   │  (Static    │    │  (Node.js /  │   │ (Managed   │  │
│   │   Assets)   │    │   Python)    │   │  RDS/Cloud │  │
│   └──────┬──────┘    └──────┬───────┘   │  SQL)      │  │
│          │                  │            └────────────┘  │
│          │    ┌─────────────┘                            │
│          │    │                                          │
│   ┌──────▼────▼──┐                                      │
│   │  Load        │                                      │
│   │  Balancer    │                                      │
│   └──────────────┘                                      │
│                                                          │
│   Current:  Vercel (Frontend Hosting)                    │
└──────────────────────────────────────────────────────────┘
```

**Current deployment:** Frontend hosted on [Vercel](https://v0-remix-of-untitled-chat-ten-vert.vercel.app/)

---

## 7. Technology Choices & Rationale

| Decision | Choice | Rationale |
|---|---|---|
| Frontend Framework | React.js | Component-based, rich ecosystem, fast rendering |
| Backend Runtime | Node.js / Python | Lightweight, non-blocking I/O, extensive library support |
| Database | PostgreSQL / MySQL | Reliable RDBMS, ACID compliance, strong query capabilities |
| Authentication | JWT | Stateless, scalable, works well with SPAs |
| Deployment | Cloud + Vercel | Auto-scaling, global CDN, zero-config deployments |

---

## 8. Scalability Considerations

| Aspect | Strategy |
|---|---|
| **Horizontal Scaling** | Stateless services behind a load balancer; add instances as traffic grows |
| **Database Scaling** | Read replicas for reporting queries; connection pooling for write-heavy loads |
| **Caching** | Redis layer for dashboard data and frequently accessed inventory counts |
| **Async Processing** | Background job queues for report generation and email notifications |
| **Modular Services** | Each service can be independently deployed and scaled |

---

## 9. Future Architecture Enhancements

| Enhancement | Description |
|---|---|
| **Microservices Migration** | Split monolithic backend into containerized microservices |
| **Event-Driven Architecture** | Introduce message queues (RabbitMQ / Kafka) for inter-service communication |
| **IoT Integration** | Real-time machine data ingestion via MQTT / WebSockets |
| **AI/ML Pipeline** | Dedicated model serving infrastructure for forecasting and optimization |
| **Mobile Backend** | GraphQL API layer for efficient mobile data fetching |

---

<p align="center">
  <em>Architecture designed for simplicity today and scalability tomorrow.</em>
</p>
