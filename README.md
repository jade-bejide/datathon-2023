# datathon-2023

[![Team Name](https://img.shields.io/badge/%E2%9A%A0%EF%B8%8F-work%20in%20progress-blueviolet)](https://img.shields.io/badge/%E2%9A%A0%EF%B8%8F-work%20in%20progress-blueviolet)

## Table of Contents ##
- [Objective](#objective)
- [Relevance](#relevance)
- [Data](#data)
- [Target](#target)

---

## Objective

Predict whether a fatal/serious casualty occurs using road traffic data about the casualty, accident, and vehicles involved.

---

## Relevance

- LV= wants to identify fatal/serious casualties early to ensure a quick response is given to those affected.

- Further to this, LV= wants to know the most important characteristics of serious casualties to help customers stay safe on the road.

---

## Data

### 1. Casualty `(primary)`

Description:
- Data on the person(s) involved in the casualty

Found in [casualty_train.csv](casualty_train.csv)


---

### 2. Vehicle `(secondary)`

Description:
- Data on the vehicles involved in the casualty

Found in [vehicle_train.csv](vehicle_train.csv)

---

### 3. Data Dictionary `(reference)`

Description:
- Field descriptions and value mappings

Found in docs/[DatathonReference.xlsx](DatathonReference.xlsx)

---

## Target

Whether `casualty_severity` in the casualty dataset is fatal (1) or serious (2), where (3) is slight.

_**Note:** Will need to map target to binary 1 (fatal, serious) and 0 (slight)._

---