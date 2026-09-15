"use strict";

const form = document.querySelector("#estimate-form");
const modelInput = document.querySelector("#model");
const yearInput = document.querySelector("#year");
const mileageInput = document.querySelector("#mileage");
const ratingInput = document.querySelector("#rating");
const button = document.querySelector("#estimate-button");
const status = document.querySelector("#status");
const estimate = document.querySelector("#estimate");

let artifact;

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function calculate() {
  const year = Number(yearInput.value);
  const mileage = Number(mileageInput.value);
  const rating = Number(ratingInput.value);
  const modelIndex = artifact.model.models.indexOf(modelInput.value);
  const [minYear, maxYear] = artifact.bounds.year;
  const [, maxMileage] = artifact.bounds.mileage;

  if (modelIndex < 0) throw new Error("Choose a Mercedes-Benz model from the dataset.");
  if (!Number.isInteger(year) || year < minYear || year > maxYear) {
    throw new Error(`Model year must be a whole number from ${minYear} to ${maxYear}.`);
  }
  if (!Number.isFinite(mileage) || mileage < 0 || mileage > maxMileage) {
    throw new Error(`Mileage must be between 0 and ${maxMileage.toLocaleString("en-US")}.`);
  }
  if (!Number.isFinite(rating) || rating < 0 || rating > 5) {
    throw new Error("Dealer rating must be between 0 and 5.");
  }

  const numeric = [year, mileage, rating];
  const numericContribution = numeric.reduce((total, value, index) => {
    const standardized = (value - artifact.model.numeric_mean[index]) / artifact.model.numeric_scale[index];
    return total + standardized * artifact.model.numeric_coefficients[index];
  }, 0);
  const value = Math.max(
    0,
    artifact.model.intercept + numericContribution + artifact.model.model_coefficients[modelIndex],
  );

  estimate.textContent = usd.format(value);
  setStatus("Estimate updated from the saved model.");
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  try {
    calculate();
  } catch (error) {
    estimate.textContent = "—";
    setStatus(error.message, true);
  }
});

fetch("./model.json")
  .then((response) => {
    if (!response.ok) throw new Error(`Model request failed with status ${response.status}.`);
    return response.json();
  })
  .then((payload) => {
    artifact = payload;
    modelInput.replaceChildren(
      ...artifact.model.models.map((model) => {
        const option = document.createElement("option");
        option.value = model;
        option.textContent = model;
        return option;
      }),
    );
    modelInput.value = artifact.defaults.model;
    yearInput.value = artifact.defaults.year;
    mileageInput.value = artifact.defaults.mileage;
    ratingInput.value = artifact.defaults.rating;
    yearInput.min = artifact.bounds.year[0];
    yearInput.max = artifact.bounds.year[1];
    mileageInput.max = artifact.bounds.mileage[1];
    document.querySelector("#row-count").textContent = artifact.dataset.rows.toLocaleString("en-US");
    document.querySelector("#mae").textContent = usd.format(artifact.evaluation.mae_usd);
    document.querySelector("#data-date").textContent = artifact.dataset.source_last_updated;
    modelInput.disabled = false;
    button.disabled = false;
    setStatus("Saved model ready.");
    calculate();
  })
  .catch((error) => {
    setStatus(`The saved model could not be loaded. ${error.message}`, true);
  });

