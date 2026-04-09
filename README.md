🏭 Dhanrakshak ERP Lite

Affordable ERP Framework for MSME Manufacturing Enterprises

<p align="center">
  <a href="https://v0-remix-of-untitled-chat-ten-vert.vercel.app/">🌐 Live Demo</a> •
  <a href="#-key-features">✨ Features</a> •
  <a href="TECH_STACK.md">🛠 Tech Stack</a> •
  <a href="DEPLOYMENT.md">🚀 Deployment</a> •
  <a href="#-api-reference">📡 API Reference</a>
</p>

---

## 📖 About

**Dhanrakshak ERP Lite** is a lightweight, modular Enterprise Resource Planning (ERP) solution purpose-built for **Micro, Small, and Medium Manufacturing Enterprises (MSMEs)**.

Most MSMEs cannot afford enterprise-grade ERP systems like SAP or Oracle. As a result, many manufacturing units rely on **Excel sheets, manual registers, and disconnected software tools** — leading to:

- ❌ Poor inventory tracking & frequent stockouts  
- ❌ Production inefficiencies & idle machines  
- ❌ Procurement delays & weak supplier management  
- ❌ Lack of financial transparency & cost control  

Dhanrakshak ERP Lite solves these problems with a **simple, affordable, and scalable** digital platform that covers the full manufacturing lifecycle — from raw materials to sales.

---

## ✨ Key Features

| Module | Capabilities |
|---|---|
| **📦 Inventory Management** | Real-time stock tracking, warehouse management, low-stock alerts, reorder notifications, inventory reports |
| **🏭 Production Management** | Work order creation & monitoring, production scheduling, machine utilization tracking, production reports |
| **🛒 Procurement Management** | Supplier database, purchase order generation, delivery tracking, vendor performance analysis |
| **🛍 Sales & Order Management** | Customer order entry, invoice generation, delivery tracking, sales analytics |
| **💰 Finance Dashboard** | Expense tracking, revenue & profit analysis, cash flow monitoring, financial reports |
| **📊 Reports & Analytics** | Cross-module reporting for inventory, production, sales, and financial data |
| **👥 User Management** | Secure login, role-based access control (Admin, Manager, Supervisor, etc.) |

---

## 📋 Requirements

📄 **Full details:** [REQUIREMENTS.md](REQUIREMENTS.md)

### 1. Functional Requirements

| Requirement | Description |
|---|---|
| **User Management** | Login/logout, role-based access (Admin, Manager, Supervisor, etc.) |
| **Inventory Management** | Track stock levels, update quantities, low-stock alerts, reorder management |
| **Production Management** | Create, assign, and monitor production work orders |
| **Procurement Management** | Manage supplier database, create and track purchase orders |
| **Sales & Orders** | Record customer orders, generate invoices, track deliveries |
| **Finance Dashboard** | Track expenses, revenue, profit margins, and cash flow |
| **Reports & Analytics** | Generate module-wise reports for inventory, production, and sales |

### 2. Non-Functional Requirements

| Requirement | Description |
|---|---|
| **Security** | Secure authentication (JWT), role-based access control, encrypted data |
| **Performance** | Sub-second dashboard load, concurrent multi-user support |
| **Scalability** | Designed to handle growing MSME data volumes gracefully |
| **Usability** | Clean, intuitive UI; mobile-friendly responsive design |

### 3. Technical Requirements

| Layer | Technology |
|---|---|
| **Frontend** | React.js / Web Dashboard with responsive UI components |
| **Backend** | Node.js or Python (FastAPI / Django) |
| **Database** | MySQL / PostgreSQL |
| **Authentication** | JWT-based token authentication |
| **Deployment** | Cloud-based (AWS / Azure / Google Cloud) |

### 4. Optional Smart Features (AI-Powered)

| Feature | Description |
|---|---|
| **Demand Forecasting** | Predict product demand using historical sales data |
| **Inventory Prediction** | Automatically suggest optimal reorder points |
| **Production Optimization** | Optimize machine scheduling to minimize downtime |

---

## 🛠 Tech Stack

📄 **Full details:** [TECH_STACK.md](TECH_STACK.md)

| Category | Technologies |
|---|---|
| **Frontend** | ![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white) |
| **Backend** | ![Node.js](https://img.shields.io/badge/Node.js-339933?style=flat&logo=nodedotjs&logoColor=white) ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) |
| **Database** | ![PostgreSQL](https://img.shields.io/badge/PostgreSQL-316192?style=flat&logo=postgresql&logoColor=white) ![MySQL](https://img.shields.io/badge/MySQL-005C84?style=flat&logo=mysql&logoColor=white) |
| **Auth** | ![JWT](https://img.shields.io/badge/JWT-000000?style=flat&logo=jsonwebtokens&logoColor=white) |
| **Cloud** | ![AWS](https://img.shields.io/badge/AWS-232F3E?style=flat&logo=amazonaws&logoColor=white) ![Azure](https://img.shields.io/badge/Azure-0078D4?style=flat&logo=microsoftazure&logoColor=white) ![GCP](https://img.shields.io/badge/GCP-4285F4?style=flat&logo=googlecloud&logoColor=white) |
| **Deployment** | ![Vercel](https://img.shields.io/badge/Vercel-000000?style=flat&logo=vercel&logoColor=white) |

---

## 🗄 Database Schema

### Core Tables

<details>
<summary><strong>📦 Inventory Table</strong></summary>

| Column | Description |
|---|---|
| `item_id` | Unique item identifier (PK) |
| `item_name` | Name of the inventory item |
| `category` | Item category |
| `stock_quantity` | Current stock count |
| `reorder_level` | Threshold for reorder alert |
| `warehouse_location` | Warehouse / shelf location |

</details>

<details>
<summary><strong>🏭 Production Table</strong></summary>

| Column | Description |
|---|---|
| `production_id` | Unique production record ID (PK) |
| `work_order_id` | Associated work order (FK) |
| `machine_id` | Machine assigned (FK) |
| `production_status` | Status (Pending / In Progress / Completed) |
| `start_time` | Production start timestamp |
| `completion_time` | Production completion timestamp |

</details>

<details>
<summary><strong>🛒 Supplier Table</strong></summary>

| Column | Description |
|---|---|
| `supplier_id` | Unique supplier ID (PK) |
| `supplier_name` | Supplier company name |
| `contact_details` | Phone / email / address |
| `supplier_rating` | Performance rating |

</details>

<details>
<summary><strong>🛍 Orders Table</strong></summary>

| Column | Description |
|---|---|
| `order_id` | Unique order ID (PK) |
| `customer_id` | Customer reference (FK) |
| `product_id` | Product ordered (FK) |
| `order_date` | Date of order placement |
| `delivery_status` | Status (Pending / Shipped / Delivered) |

</details>

<details>
<summary><strong>💰 Finance Table</strong></summary>

| Column | Description |
|---|---|
| `transaction_id` | Unique transaction ID (PK) |
| `transaction_type` | Income / Expense |
| `amount` | Transaction amount |
| `date` | Transaction date |
| `description` | Transaction description |

</details>

---

## 📡 API Reference

### Inventory APIs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/inventory` | List all inventory items |
| `POST` | `/inventory/add` | Add a new inventory item |
| `PUT` | `/inventory/update` | Update an existing item |
| `DELETE` | `/inventory/remove` | Remove an inventory item |

### Production APIs

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/production/workorder` | Create a new work order |
| `GET` | `/production/status` | Get production status |
| `PUT` | `/production/update` | Update production record |

### Procurement APIs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/suppliers` | List all suppliers |
| `POST` | `/purchase-order` | Create a purchase order |
| `PUT` | `/purchase-order/update` | Update a purchase order |

### Sales APIs

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/orders` | List all orders |
| `POST` | `/orders/create` | Create a new order |
| `PUT` | `/orders/update` | Update an order |

---

## 🚀 Getting Started

📄 **Full deployment guide:** [DEPLOYMENT.md](DEPLOYMENT.md)

### Prerequisites

- **Node.js** ≥ 18.x  
- **npm** ≥ 9.x  
- **PostgreSQL** or **MySQL** database instance  

### Installation

```bash
# Clone the repository
git clone https://github.com/Harsharyan05/Sharpaithon-2.0.git
cd Sharpaithon-2.0

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env
# Edit .env with your database credentials and JWT secret

# Run database migrations
npm run migrate

# Start the development server
npm run dev
```

The application will be available at `http://localhost:3000`.

### 🌐 Live Deployment

The application is deployed and accessible at:  
**🔗 [https://v0-remix-of-untitled-chat-ten-vert.vercel.app/](https://v0-remix-of-untitled-chat-ten-vert.vercel.app/)**

---

## 👥 Target Users

| User Type | Role |
|---|---|
| **Factory Owners** | Full system access, financial overview, strategic decisions |
| **Operations Managers** | Production planning, inventory oversight, procurement |
| **Production Supervisors** | Work order management, machine monitoring |
| **Inventory Managers** | Stock tracking, reorder management, warehouse operations |
| **Procurement Teams** | Supplier management, purchase order processing |
| **Finance Teams** | Expense tracking, revenue analysis, financial reporting |

---

## 🔮 Future Roadmap

- [ ] 🤖 AI-driven analytics & predictive insights  
- [ ] 📱 Native mobile applications (iOS & Android)  
- [ ] 🔌 IoT-based machine monitoring integration  
- [ ] 🧾 Integration with accounting systems (Tally, Zoho Books)  
- [ ] 🔗 Advanced supply chain management module  
- [ ] 📈 Enhanced reporting with exportable dashboards  

---

