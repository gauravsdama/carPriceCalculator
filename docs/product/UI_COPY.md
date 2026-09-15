# Car Price Calculator UI copy

Human-owned source of truth for visible and accessibility text in the Flask app and the static saved-model demo. Dynamic values are shown in braces. Wording marked `approved` must remain exact.

| ID | Surface / state | Exact text | Status | Implementation |
|---|---|---|---|---|
| `app.meta.title` | Flask page | Mercedes-Benz Price Calculator | inventory | `mercedesbenzRIDGE.py` |
| `app.brand` | Flask header | Car Price Calculator | inventory | `mercedesbenzRIDGE.py` |
| `app.snapshot` | Flask header | {listing count} saved listings · Random Forest | inventory | `mercedesbenzRIDGE.py` |
| `app.heading` | Flask page | Explore a dataset-based listing estimate. | inventory | `mercedesbenzRIDGE.py` |
| `app.intro` | Flask page | This local demo trains on a saved snapshot of Mercedes-Benz asking prices. It is useful for exploring the model, not valuing a specific vehicle. | inventory | `mercedesbenzRIDGE.py` |
| `app.metrics` | Flask evidence | Model summary; Listings; Model R2; RMSE; Dataset range; Year range; Mileage range; Median price | inventory | `mercedesbenzRIDGE.py` |
| `app.fields` | Flask form | Mercedes-Benz model; Model year; Mileage (miles); Dealer rating (0–5) | inventory | `mercedesbenzRIDGE.py` |
| `app.submit` | Flask form | Estimate listing price | inventory | `mercedesbenzRIDGE.py` |
| `app.result` | Flask result | Estimated listing price; {amount}; Random Forest output from the saved dataset. Actual prices may differ. | inventory | `mercedesbenzRIDGE.py` |
| `app.disclaimer` | Flask form | Educational estimate from saved listing data—not a live valuation or buying recommendation. | approved | `mercedesbenzRIDGE.py` |
| `app.error.number` | Flask error | Use numbers for model year, mileage, and dealer rating. | inventory | `mercedesbenzRIDGE.py` |
| `app.error.model` | Flask error | Choose a Mercedes-Benz model from the dataset. | inventory | `mercedesbenzRIDGE.py` |
| `app.error.year` | Flask error | Model year must be a whole number from {minimum} to {maximum}. | inventory | `mercedesbenzRIDGE.py` |
| `app.error.mileage` | Flask error | Mileage must be between 0 and {maximum}. | inventory | `mercedesbenzRIDGE.py` |
| `app.error.rating` | Flask error | Dealer rating must be between 0 and 5. | inventory | `mercedesbenzRIDGE.py` |
| `static.meta.title` | Static page | Car Price Calculator · Saved model demo | inventory | `docs/demo/index.html` |
| `static.brand` | Static header | Car Price Calculator; Car Price Calculator home; Saved model demo | inventory | `docs/demo/index.html` |
| `static.heading` | Static page | Test a listing estimate in your browser. | inventory | `docs/demo/index.html` |
| `static.intro` | Static page | This static demo runs a compact Ridge model trained on a saved snapshot of Mercedes-Benz asking prices. Nothing is sent to a server. | inventory | `docs/demo/index.html` |
| `static.evidence` | Static evidence | Saved model evidence; Listings; Holdout MAE; Data updated | inventory | `docs/demo/index.html` |
| `static.presets` | Static examples | Five quick starts; Load a saved vehicle example, then change any input.; GLC 300; C-Class C 300; E-Class E 350 4MATIC; GLE 350 4MATIC; S-Class S 580 4MATIC | inventory | `docs/demo/index.html` |
| `static.fields` | Static form | Mercedes-Benz model; Loading models…; Model year; Dealer rating (0–5); Mileage (miles) | inventory | `docs/demo/index.html` |
| `static.submit` | Static form | Estimate listing price | inventory | `docs/demo/index.html` |
| `static.status` | Static states | Loading saved model…; Saved model ready.; Estimate updated from the saved model.; The saved model could not be loaded. {error} | inventory | `docs/demo/index.html`, `docs/demo/app.js` |
| `static.errors` | Static errors | Choose a Mercedes-Benz model from the dataset.; Model year must be a whole number from {minimum} to {maximum}.; Mileage must be between 0 and {maximum}.; Dealer rating must be between 0 and 5. | inventory | `docs/demo/app.js` |
| `static.result` | Static result | Estimated listing price; {amount}; Ridge snapshot output from the saved dataset. The local Flask app uses a separate Random Forest model. | inventory | `docs/demo/index.html` |
| `static.limitations` | Static notes | Read the estimate correctly | inventory | `docs/demo/index.html` |
| `static.disclaimer` | Static notes | Educational estimate from saved listing data—not a live valuation or buying recommendation. | approved | `docs/demo/index.html` |

Long-form method, limitation, and dataset-attribution sentences under `static.limitations` are factual documentation displayed in the UI. Update this inventory whenever those statements change.
