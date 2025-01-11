function goToPage2() {
  window.location.href = "page2.html";
}

function plotCreativity(event) {
  event.preventDefault();
  alert("Plotting the creativity");
}

function calculatesimilarity(event) {
  event.preventDefault();
  alert("Plotting the visual creativity");
}

function analyzeData(event) {
  event.preventDefault();

  // Collect the word list from the textarea
  const wordList = document.querySelector('textarea[name="wordlist"]').value;

  // Prepare the data to be sent to the backend
  const formData = { wordList };

  fetch("/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  })
    .then((response) => response.json())
    .then((data) => {
      console.log("Analysis result:", data);
      alert("Data analyzed. Check console for results.");
    })
    .catch((error) => {
      console.error("Error:", error);
    });
}

function uploadAndAnalyze(event) {
  event.preventDefault();

  const fileInput = document.querySelector('input[type="file"]');
  const file = fileInput.files[0];

  if (file) {
    const formData = new FormData();
    formData.append("file", file);

    fetch("/upload", {
      method: "POST",
      body: formData,
    })
      .then((response) => response.json())
      .then((data) => {
        console.log("Upload and analysis result:", data);
        alert("File uploaded and data analyzed. Check console for results.");
      })
      .catch((error) => {
        console.error("Error:", error);
      });
  } else {
    alert("Please select a file to upload.");
  }
}

// JavaScript to update the word1 value based on the dropdown selection
