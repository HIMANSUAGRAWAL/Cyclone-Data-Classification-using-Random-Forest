// static/js/script.js
document.getElementById("cycloneForm").addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent the form from refreshing the page

    let max_wind = document.getElementById("max_wind").value;
    let min_pressure = document.getElementById("min_pressure").value;
    let latitude = document.getElementById("latitude").value;
    let year = document.getElementById("year").value;
    let longitude = document.getElementById("longitude").value;

    fetch('/classify', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            max_wind: max_wind,
            min_pressure: min_pressure,
            latitude: latitude,
            year: year,
            longitude: longitude
        })
    })
        .then(response => response.json())
        .then(data => {
            document.getElementById("result").innerText = "Predicted Cyclone Type: " + data.prediction + "\nDescription: " + data.description;
        })
        .catch((error) => {
            console.error('Error:', error);
        });
});
