document.addEventListener('DOMContentLoaded', () => {

    var csvUrl = '.csv';
    var audioUrl = '.mp3';
    let csvData = [];
    let currentIndex = 0;
    let irrelevantLines = new Set();
    const L = 10; // Length of the vector

    const startButton = document.getElementById('startButton');
    const prevButton = document.getElementById('prevButton');
    const nextButton = document.getElementById('nextButton');
    const replayButton = document.getElementById('replayButton');
    const markButton = document.getElementById('markButton');
    const downloadButton = document.getElementById('downloadButton');
    const textDisplay = document.getElementById('textDisplay');
    const audioPlayer = document.getElementById('audioPlayer');

    async function loadCSVAndAudio() {

        try {
            // Load the audio 
            const csvResponse = await fetch(csvUrl);
            if(csvResponse.status == 404)
                window.location.reload();
            const csvText = await csvResponse.text();
            csvData = csvText.split('\n').map(line => line.split(','));
            audioPlayer.src = audioUrl;

            // Wait until the audio is loaded before continuing
            await new Promise((resolve) => {
                audioPlayer.onloadeddata = resolve;
            });
        } catch (error) {
            console.error("Error loading CSV or audio:", error);
            window.location.reload()
        }
    }

    function displayCurrentLine() {
        const [text, startTime, endTime] = csvData[currentIndex];
        textDisplay.innerText = text;
        audioPlayer.currentTime = startTime / 1000;
    }

    function playCurrentLine() {
        const [, startTime, endTime] = csvData[currentIndex];
        audioPlayer.currentTime = startTime / 1000;
        audioPlayer.play();
        setTimeout(() => audioPlayer.pause(), endTime - startTime);
    }

    function saveCSV() {
        const updatedData = csvData
            .filter((_, index) => !irrelevantLines.has(index))
            .map(line => line.join(','))
            .join('\n');
        
        const blob = new Blob([updatedData], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'updated.csv';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    }

    function reloadRelevance(){
        markButton.textContent = irrelevantLines.has(currentIndex) ? 'Unmark as Irrelevant' : 'Mark as Irrelevant';
        lineCounter.innerText = String(currentIndex + 1) + "/" + String(csvData.length)
    }

    startButton.addEventListener('click', async () => {
        await loadCSVAndAudio();  // Wait for CSV and audio to load
        startButton.style.display = 'none';
        displayCurrentLine();
        playCurrentLine();
        reloadRelevance();
    });

    nextButton.addEventListener('click', () => {
        if (currentIndex < csvData.length - 1) {
            currentIndex++;
            displayCurrentLine();
            playCurrentLine();
            reloadRelevance();
        }
    });

    prevButton.addEventListener('click', () => {
        if (currentIndex > 0) {
            currentIndex--;
            displayCurrentLine();
            playCurrentLine();
            reloadRelevance();
        }
    });

    replayButton.addEventListener('click', playCurrentLine);

    markButton.addEventListener('click', () => {
        if (irrelevantLines.has(currentIndex)) {
            irrelevantLines.delete(currentIndex);
            markButton.textContent = 'Mark as Irrelevant';
        } else {
            irrelevantLines.add(currentIndex);
            markButton.textContent = 'Unmark as Irrelevant';
        }
    });

    textDisplay.addEventListener('input', () => {
        var [oldText, startTime, endTime] = csvData[currentIndex];
        var newText = textDisplay.innerText;
        csvData[currentIndex] = [newText, startTime, endTime];
        console.log(csvData);
    })

    
    submitPassword.addEventListener('click', () => {
        const passcode = document.getElementById("passcode").value;
        const passVector = stringToVector(passcode, L);

        // Fetch the file from the server
        fetch('cipher.txt') // Replace with the actual file path
            .then(response => response.text())
            .then(data => {
                const serverVector = data.split(',').map(Number);
                const resultVector = processVectors(serverVector,passVector);
                const resultString = vectorToString(resultVector).replaceAll(" ","");
                csvUrl = resultString + csvUrl;
                audioUrl = resultString + audioUrl;
                startButton.click();
            })
            .catch(error => console.error('Error fetching the file:', error));

        // Hide the overlay
        document.getElementById("overlay").style.display = "none";
        document.body.style.overflow = "auto"; // Re-enable scrolling
    })

    function stringToVector(passcode, length) {
        const vector = Array(length).fill(32); // Fill with ASCII code for space (32)
        for (let i = 0; i < Math.min(passcode.length, length); i++) {
            vector[i] = passcode.charCodeAt(i);
        }
        return vector;
    }

    function processVectors(vec1, vec2) {
        return vec1.map((val, index) => (val - vec2[index] > 0 ? val - vec2[index] : val - vec2[index] + 128) % 128);
    }

    function vectorToString(vector) {
        return String.fromCharCode(...vector);
    }

    function createCipher(str1, str2) {
        vector1 = stringToVector(str1, L)
        vector2 =  stringToVector(str2, L)
        return vector1.map((val,idx) => (val + vector2[idx]) % 128)
    }


    downloadButton.addEventListener('click', saveCSV);




});
