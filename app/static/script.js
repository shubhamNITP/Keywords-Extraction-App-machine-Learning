// -------- DARK MODE --------
const toggleDark = document.getElementById("toggleDark");
toggleDark.addEventListener("click", () => {
    document.body.classList.toggle("dark");
    document.body.classList.toggle("light");

    toggleDark.textContent = document.body.classList.contains("dark")
        ? "☀ Light Mode"
        : "🌙 Dark Mode";
});

// -------- FILE UPLOAD --------
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const fileList = document.getElementById("fileList");

dropzone.addEventListener("click", () => fileInput.click());

dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.style.borderColor = "#4e6bff";
});

dropzone.addEventListener("dragleave", () => {
    dropzone.style.borderColor = "#888";
});

dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    fileInput.files = e.dataTransfer.files;
    showFiles();
});

fileInput.addEventListener("change", showFiles);

function showFiles() {
    fileList.innerHTML = "";
    [...fileInput.files].forEach(f => {
        fileList.innerHTML += `<div>${f.name}</div>`;
    });
}

// -------- EXTRACT KEYWORDS --------
const extractBtn = document.getElementById("extractBtn");
const loader = document.getElementById("loader");
const resultsDiv = document.getElementById("results");

extractBtn.onclick = async () => {

    loader.classList.remove("hidden");
    resultsDiv.innerHTML = "";

    const formData = new FormData();
    [...fileInput.files].forEach(f => formData.append("pdf_file", f));

    formData.append("text_input", document.getElementById("text_input").value);
    formData.append("top_k", document.getElementById("top_k").value);

    const res = await fetch("/extract", { method: "POST", body: formData });
    const data = await res.json();

    loader.classList.add("hidden");

    renderResults(data.results);
};

function renderResults(all) {
    resultsDiv.innerHTML = "";

    all.forEach(item => {
        let html = `
        <div class="result-block">
            <h3>${item.filename}</h3>
            <h4>Top Keywords:</h4>
        `;

        item.keywords.forEach(k => {
            html += `<div class="keyword">${k.word} — <b>${k.score}</b></div>`;
        });

        html += `</div>`;
        resultsDiv.innerHTML += html;
    });
}

// -------- EXPORT CSV --------
document.getElementById("exportBtn").onclick = () => {
    if (!resultsDiv.innerText.trim()) return alert("No results to export!");

    let csv = "file,keyword,score\n";

    document.querySelectorAll(".result-block").forEach(block => {
        const file = block.querySelector("h3").innerText;

        block.querySelectorAll(".keyword").forEach(k => {
            const [word, score] = k.innerText.split(" — ");
            csv += `${file},${word},${score}\n`;
        });
    });

    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "keywords.csv";
    a.click();
};
