async function predictWord() {
    let text = document.getElementById("user_input").value;
    let response = await fetch("/predict", {
        method: "POST",
        headers: {"Content-Type" : "application/json"},
        body: JSON.stringify({text : text, num_words: 1})
    });
    let data = await response.json();
    document.getElementById("predictedResult").innerText = data.predicted_text;
}

async function findMeaning() {
    let word = document.getElementById("meaningInput").value;
    let response = await fetch("/meaning", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ word: word })
    });
    let data = await response.json();
    document.getElementById("meaningResult").innerText = data.meaning;
}

async function correctSentence() {
    let sentence = document.getElementById("correctInput").value;
    let response = await fetch("/correct", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sentence: sentence })
    });
    let data = await response.json();
    document.getElementById("correctedResult").innerText = data.corrected_sentence;
}
