document.addEventListener('DOMContentLoaded', function() {
    const predictionForm = document.getElementById('predictionForm');
    if (predictionForm) {
        predictionForm.addEventListener('submit', handlePredictionSubmit);
    }
});

function filterSymptoms() {
    let input = document.getElementById('symptomSearch').value.toLowerCase();
    let items = document.getElementsByClassName('symptom-item');
    
    for (let i = 0; i < items.length; i++) {
        let label = items[i].getElementsByTagName('label')[0].innerText.toLowerCase();
        if (label.indexOf(input) > -1) {
            items[i].style.display = "";
        } else {
            items[i].style.display = "none";
        }
    }
}

function handlePredictionSubmit(e) {
    e.preventDefault();
    
    const btn = document.getElementById('predictBtn');
    btn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing Symptoms...';
    btn.disabled = true;

    let selectedSymptoms = [];
    document.querySelectorAll('.symptom-checkbox:checked').forEach((box) => {
        selectedSymptoms.push(box.value);
    });

    fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ symptoms: selectedSymptoms })
    })
    .then(response => response.json())
    .then(data => {
        btn.innerHTML = 'Analyze Symptoms';
        btn.disabled = false;

        if (data.error) {
            alert("Error: " + data.error);
            return;
        }

        document.getElementById('placeholderCard').style.display = 'none';
        document.getElementById('resultCard').style.display = 'block';

        document.getElementById('diseaseResult').innerText = data.disease;
        document.getElementById('confidenceText').innerText = data.confidence + '%';
        document.getElementById('confidenceBar').style.width = data.confidence + '%';
        document.getElementById('llmText').innerText = data.llm_explanation.replace(/\*/g, '');
        document.getElementById('downloadPdfBtn').href = "/download_pdf/" + data.history_id;

        const shapList = document.getElementById('shapList');
        shapList.innerHTML = ''; 
        if (data.contributing_factors && data.contributing_factors.length > 0) {
            data.contributing_factors.forEach(factor => {
                shapList.innerHTML += `
                    <li class="list-group-item d-flex justify-content-between align-items-center bg-transparent px-0">
                        <span><i class="text-danger me-2">•</i> ${factor.symptom}</span>
                        <span class="badge bg-secondary rounded-pill">High Impact</span>
                    </li>`;
            });
        } else {
            shapList.innerHTML = '<li class="list-group-item text-muted bg-transparent px-0">No specific dominant factors found.</li>';
        }
    })
    .catch(err => {
        console.error('Error:', err);
        alert("An error occurred while connecting to the AI system.");
        btn.innerHTML = 'Analyze Symptoms';
        btn.disabled = false;
    });
}

function findHospitals() {
    const btn = document.getElementById('hospitalBtn');
    const originalText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-border spinner-border-sm"></span> Locating...';
    btn.disabled = true;

    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                const lat = position.coords.latitude;
                const lon = position.coords.longitude;
                window.open(`https://www.google.com/maps/search/hospitals/@${lat},${lon},14z`, '_blank');
                btn.innerHTML = originalText;
                btn.disabled = false;
            },
            (error) => {
                window.open('https://www.google.com/maps/search/hospitals+near+me', '_blank');
                btn.innerHTML = originalText;
                btn.disabled = false;
            }
        );
    } else {
        window.open('https://www.google.com/maps/search/hospitals+near+me', '_blank');
        btn.innerHTML = originalText;
        btn.disabled = false;
    }
}