"use strict";

const form = document.querySelector("#estimate-form");
const modelInput = document.querySelector("#model");
const yearInput = document.querySelector("#year");
const mileageInput = document.querySelector("#mileage");
const ratingInput = document.querySelector("#rating");
const button = document.querySelector("#estimate-button");
const status = document.querySelector("#status");
const estimate = document.querySelector("#estimate");
const presets = [...document.querySelectorAll(".preset")];

let artifact;
const artifactType = "car-price-calculator.static-model";
const schemaVersion = 1;
const maxArtifactBytes = 1_000_000;
const modelTimeoutMs = 5_000;

const usd = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

function setStatus(message, isError = false) {
  status.textContent = message;
  status.classList.toggle("error", isError);
}

function requireArtifact(condition, message) {
  if (!condition) throw new Error(`Saved model is invalid: ${message}.`);
}

function isFiniteNumber(value) {
  return typeof value === "number" && Number.isFinite(value);
}

function isNumericArray(values, length) {
  return Array.isArray(values) && values.length === length && values.every(isFiniteNumber);
}

function validateArtifact(payload) {
  requireArtifact(payload && typeof payload === "object" && !Array.isArray(payload), "expected an object");
  requireArtifact(payload.artifact_type === artifactType, "unsupported artifact type");
  requireArtifact(payload.schema_version === schemaVersion, "unsupported schema version");

  const { bounds, dataset, defaults, evaluation, model } = payload;
  requireArtifact(bounds && dataset && defaults && evaluation && model, "missing required fields");
  requireArtifact(Number.isInteger(dataset.rows) && dataset.rows > 0, "dataset row count");
  requireArtifact(typeof dataset.source_last_updated === "string", "dataset update date");
  requireArtifact(isFiniteNumber(evaluation.mae_usd) && evaluation.mae_usd >= 0, "evaluation MAE");
  requireArtifact(model.method === "Ridge regression", "unsupported model method");
  requireArtifact(
    JSON.stringify(model.features) === JSON.stringify(["Year", "Mileage", "Rating", "Model"]),
    "feature order",
  );
  requireArtifact(
    JSON.stringify(model.numeric_features) === JSON.stringify(["Year", "Mileage", "Rating"]),
    "numeric feature order",
  );
  requireArtifact(isFiniteNumber(model.intercept), "model intercept");
  requireArtifact(isNumericArray(model.numeric_mean, 3), "numeric means");
  requireArtifact(isNumericArray(model.numeric_scale, 3), "numeric scales");
  requireArtifact(model.numeric_scale.every((value) => value > 0), "numeric scales");
  requireArtifact(isNumericArray(model.numeric_coefficients, 3), "numeric coefficients");
  requireArtifact(
    Array.isArray(model.models) && model.models.length > 0 && model.models.length <= 1_000,
    "model names",
  );
  requireArtifact(
    model.models.every((value) => typeof value === "string" && value.length > 0 && value.length <= 100),
    "model names",
  );
  requireArtifact(new Set(model.models).size === model.models.length, "duplicate model names");
  requireArtifact(
    isNumericArray(model.model_coefficients, model.models.length),
    "model coefficients",
  );

  for (const name of ["year", "mileage", "rating", "price"]) {
    requireArtifact(
      isNumericArray(bounds[name], 2) && bounds[name][0] <= bounds[name][1],
      `${name} bounds`,
    );
  }
  requireArtifact(model.models.includes(defaults.model), "default model");
  for (const name of ["year", "mileage", "rating"]) {
    requireArtifact(
      isFiniteNumber(defaults[name]) && defaults[name] >= bounds[name][0] && defaults[name] <= bounds[name][1],
      `default ${name}`,
    );
  }
  return payload;
}

async function loadArtifact() {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), modelTimeoutMs);
  try {
    const response = await fetch("./model.json", { signal: controller.signal });
    if (!response.ok) throw new Error(`Model request failed with status ${response.status}.`);
    const declaredLength = Number(response.headers.get("content-length"));
    if (Number.isFinite(declaredLength) && declaredLength > maxArtifactBytes) {
      throw new Error("Saved model exceeds the size limit.");
    }
    const text = await response.text();
    if (text.length > maxArtifactBytes) throw new Error("Saved model exceeds the size limit.");
    try {
      return validateArtifact(JSON.parse(text));
    } catch (error) {
      if (error instanceof SyntaxError) throw new Error("Saved model is not valid JSON.");
      throw error;
    }
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Model request timed out.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
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
  setStatus("Estimate updated.");
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

presets.forEach((preset) => {
  preset.addEventListener("click", () => {
    modelInput.value = preset.dataset.model;
    yearInput.value = preset.dataset.year;
    mileageInput.value = preset.dataset.mileage;
    ratingInput.value = preset.dataset.rating;
    presets.forEach((item) => item.classList.toggle("active", item === preset));
    calculate();
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    form.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
  });
});

loadArtifact()
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
    presets.forEach((preset) => {
      preset.disabled = false;
    });
    setStatus("Calculator ready.");
    calculate();
  })
  .catch((error) => {
    setStatus(`The saved model could not be loaded. ${error.message}`, true);
  });
