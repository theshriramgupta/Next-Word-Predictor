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
