# TODO: Implement Company Profile Page for Bank Kuy

- [x] Add new route `/company` in `app.py` to render `company.html`
- [x] Create `templates/company.html` with multiple sections (About, Services, Why Choose Us, Contact) including floating chatbot widget
- [ ] Update `templates/index.html` to add navigation link to company profile page
- [ ] Update `static/css/style.css` to add styles for company page sections and fade-in animations
- [ ] Update `static/js/script.js` to add Intersection Observer for scroll-based fade-in animations
- [ ] Test the new `/company` route and page loading
- [ ] Verify scroll-based fade-in animations on page sections
- [ ] Ensure floating chatbot widget functions on company page
- [ ] Check responsiveness and styling

# TODO: Implement Result Page for Loan Prediction

- [x] Create new page `pages/3_Result.py` to display prediction results (Streamlit version)
- [x] Add "Coba Kembali" button to return to predict page
- [x] Add "Tanya AI" button that appears only for rejected predictions
- [x] Integrate LLM call for "Tanya AI" to explain rejection reasons
- [x] Modify `pages/1_Predict.py` to store prediction data in session state and navigate to result page
- [x] Create `templates/result.html` for Flask web app
- [x] Modify Flask `app.py` to redirect to result page after prediction
- [x] Add `/ask_ai` endpoint for AJAX LLM calls
- [x] Fix model file paths in Flask app.py
- [x] Test Flask app startup and basic functionality
