const uploadView = document.getElementById("uploadView");
const loadingView = document.getElementById("loadingView");
const resultView = document.getElementById("resultView");
const fileInput = document.getElementById("fileInput");
const uploadBtn = document.getElementById("uploadBtn");
const dropZone = document.getElementById("dropZone");
const fileName = document.getElementById("fileName");
const errorBox = document.getElementById("error");

uploadBtn.addEventListener("click", e => { e.stopPropagation(); fileInput.click(); });
dropZone.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files.length) analyze(fileInput.files[0]); });

["dragenter", "dragover"].forEach(evt => dropZone.addEventListener(evt, e => {
    e.preventDefault(); dropZone.classList.add("dragover");
}));
["dragleave", "drop"].forEach(evt => dropZone.addEventListener(evt, e => {
    e.preventDefault(); dropZone.classList.remove("dragover");
}));
dropZone.addEventListener("drop", e => {
    if (e.dataTransfer.files.length) analyze(e.dataTransfer.files[0]);
});

function metricClass(value) {
    return value >= 68 ? "bad" : value >= 42 ? "warn" : "good";
}

function renderForensics(forensics) {
    const grid = document.getElementById("forensicGrid");
    grid.innerHTML = "";
    const labels = [
        ["noise_inconsistency", "Local noise inconsistency"],
        ["ela_anomaly", "Recompression / ELA anomaly"],
        ["clone_similarity", "Copy-move / clone similarity"],
        ["text_region_anomaly", "Text-region anomaly"],
        ["software_signal", "Editing software metadata"]
    ];
    labels.forEach(([key, label]) => {
        const value = Number(forensics[key] || 0);
        const div = document.createElement("div");
        div.className = `forensic-item ${metricClass(value)}`;
        div.innerHTML = `<span>${label}</span><strong>${value.toFixed(1)}%</strong>`;
        grid.appendChild(div);
    });
}

async function analyze(file) {
    errorBox.classList.add("hidden");
    fileName.textContent = file.name;
    const data = new FormData();
    data.append("image", file);

    uploadView.classList.add("hidden");
    resultView.classList.add("hidden");
    loadingView.classList.remove("hidden");

    try {
        const response = await fetch("/analyze", { method: "POST", body: data });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || "Analysis failed.");

        document.getElementById("verdict").textContent = result.verdict;
        document.getElementById("explanation").textContent = result.explanation;
        document.getElementById("documentConfidence").textContent = `${result.document_confidence}%`;
        document.getElementById("tamperProbability").textContent = `${result.tampering_probability}%`;
        document.getElementById("aiProbability").textContent = `${result.ai_probability}%`;
        document.getElementById("scoreBar").style.width = `${result.is_document ? result.tampering_probability : result.ai_probability}%`;

        const card = document.getElementById("verdictCard");
        card.className = `result-card ${result.level}`;
        document.getElementById("level").textContent =
            result.level === "low" ? "Low concern" :
            result.level === "medium" ? "Review recommended" : "High concern";

        document.getElementById("modeText").textContent = result.is_document
            ? "Document forensic mode: tampering evidence is weighted more heavily than generic AI-image detection."
            : "Photo mode: AI-generation detection is the primary signal.";

        document.getElementById("originalImage").src = result.image_url;
        document.getElementById("heatmapImage").src = result.heatmap_url;
        renderForensics(result.forensics);

        const list = document.getElementById("metadataList");
        list.innerHTML = "";
        result.metadata.findings.forEach(item => {
            const li = document.createElement("li");
            li.textContent = item;
            list.appendChild(li);
        });

        loadingView.classList.add("hidden");
        resultView.classList.remove("hidden");
    } catch (err) {
        loadingView.classList.add("hidden");
        uploadView.classList.remove("hidden");
        errorBox.textContent = err.message;
        errorBox.classList.remove("hidden");
    }
}

document.getElementById("againBtn").addEventListener("click", () => {
    fileInput.value = "";
    fileName.textContent = "";
    resultView.classList.add("hidden");
    uploadView.classList.remove("hidden");
});
