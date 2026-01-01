const fs = require("fs");
const path = require("path");

const commonPath = "https://palash90.github.io/site-assets/blogs/";

// Map of variables and their replacement values
const variableMap = {
  "setting-up-a-site-part-1-home-page":
    commonPath + "setting-up-a-site/home-page.png",
  "setting-up-a-site-part-1-blog-page":
    commonPath + "setting-up-a-site/blog-page.png",
  "setting-up-a-site-part-1-proj-page":
    commonPath + "setting-up-a-site/proj-page.png",
  "setting-up-a-site-part-1-about-page":
    commonPath + "setting-up-a-site/about-page.png",
  "static-file-hosting-github-pages":
    commonPath + "static-file-hosting/github-pages-configuration.png",
  "go-tut-1-go": commonPath + "go-tut/ex/1.go",
  "non-blocking-server-architecture":
    commonPath + "fearless-rust/non-blocking-server.jpg",
  "iron-learn-7-epoch-err":
    commonPath + "iron-learn/iron-learn-7-epoch-error.png",
  "iron-learn-8-math-chits":
    commonPath + "iron-learn/iron-learn-math-chits.png",
  "iron-learn-8-actual-plot": commonPath + "iron-learn/actual-plot.png",
  "iron-learn-9-original": commonPath + "iron-learn/iron-learn-9-original.png",
  "iron-learn-9-plot_FINAL_1024":
    commonPath + "iron-learn/iron-learn-9-plot_FINAL_1024.png",
  "iron-learn-9-plot_final":
    commonPath + "iron-learn/iron-learn-9-plot_final.png",
  "iron-learn-9-plot_static":
    commonPath + "iron-learn/iron-learn-9-plot_static.png",
  "iron-learn-10-output0": commonPath + "iron-learn/output0.png",
  "iron-learn-10-output803000":
    commonPath + "iron-learn/output803000.png",
  "iron-learn-10-timeline": commonPath + "iron-learn/timeline.gif",
  "iron-learn-10-512": commonPath + "iron-learn/output-512-reconstruction.png",
};

// Function to replace patterns in a file
const replaceVariablesInFile = (filePath) => {
  // Read the file content
  fs.readFile(filePath, "utf8", (err, data) => {
    if (err) {
      console.error(`Error reading file ${filePath}: ${err}`);
      return;
    }

    // Regex to match ${some_variable} or ${some_other_variable}
    let modifiedData = data;
    Object.keys(variableMap).forEach((key) => {
      const regex = new RegExp(`\\$\\{${key}\\}`, "g"); // Match ${some_variable}
      modifiedData = modifiedData.replace(regex, variableMap[key]);
    });

    var targetFilePath = "target/" + filePath;
    const dir = path.dirname(targetFilePath);
    fs.mkdirSync(dir, { recursive: true });

    // Write the modified data back to the file
    fs.writeFile(targetFilePath, modifiedData, "utf8", (err) => {
      if (err) {
        console.error(`Error writing file ${targetFilePath}: ${err}`);
      } else {
        console.log(`File updated: ${targetFilePath}`);
      }
    });
  });
};

// Function to recursively read the directory
const processDirectory = (dirPath) => {
  fs.readdir(dirPath, { withFileTypes: true }, (err, files) => {
    if (err) {
      console.error(`Error reading directory ${dirPath}: ${err}`);
      return;
    }

    files.forEach((file) => {
      const fullPath = path.join(dirPath, file.name);

      if (file.isDirectory()) {
        console.log("Process Directory", file.name);
        // Recursively process directories
        processDirectory(fullPath);
      } else {
        // Process file
        if (path.extname(file.name) === ".md") {
          console.log("\tChanging", fullPath);
          replaceVariablesInFile(fullPath);
        }
      }
    });
  });
};

// Example: Start processing the directory
const directoryPath = "./blogs";
processDirectory(directoryPath);
