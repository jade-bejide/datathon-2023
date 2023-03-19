# datathon-2023

[![Team Name](https://img.shields.io/badge/%E2%9A%A0%EF%B8%8F-work%20in%20progress-blueviolet)](https://img.shields.io/badge/%E2%9A%A0%EF%B8%8F-work%20in%20progress-blueviolet)

## Table of Contents ##
- [datathon-2023](#datathon-2023)
    - [About the Datathon](#about-the-datathon)
    - [Team](#team)
    - [Objectives](#objectives)
    - [Dataset](#dataset)
    - [Preprocessing](#preprocessing)
    - [Challenges](#challenges)


---

## About the Datathon

This Datathon was hosted by the Bristol Data Science Society (BDSS) in association with LV=. The format of the competition was a data science oriented Hackathon where we were given a real world dataset and a prediction task.

---

## Team

Our team (called `work in progress`) consists of 2 people from the University of Bristol:
- [Nikhil Parimi](https://www.linkedin.com/in/nikhil-parimi/) (Computer Science, 2nd Year)
- [Jadesola Bejide](https://www.linkedin.com/in/jadesolabejide/) (Computer Science, 2nd Year)


---

## Objectives

Predict whether a fatal/serious casualty occurs 

Predict the `casualty_severity` column in the [casualty_test.csv](casualty_test.csv) dataset for whether an accident is fatal/serious or slight using road traffic data about the casualty, accident, and vehicles involved.

Note: Map targets to binary 1 (fatal, serious) and 0 (slight).

---

## Dataset

### 1. Casualty `(primary)`
- Data on the person(s) involved in the casualty; [casualty_train.csv](casualty_train.csv)

### 2. Vehicle `(secondary)`
- Data on the vehicles involved in the casualty; [vehicle_train.csv](vehicle_train.csv)

### 3. Data Dictionary `(reference)`
- Field descriptions and value mappings; [DatathonReference.xlsx](DatathonReference.xlsx)

---

## Preprocessing

1. Merged both csv files together on common field `accident_reference`
2. Merged the values of the column `casualty_severity` into binary form; mapping 1 to fatal / serious and 0 to slight.
3. lorem ipsum

---

## Challenges

### Challenge #1

lorem ipsum lorem ipsum


### Challenge #2

lorem ipsum lorem ipsum


### Challenge #3

lorem ipsum lorem ipsum

---