require("@nomicfoundation/hardhat-toolbox");

module.exports = {
  solidity: "0.8.28",
  networks: {
    mumbai: {
      url: "https://polygon-mumbai.infura.io/v3/YOUR_INFURA_API_KEY",
      accounts: ["62d8aead14f164f78fcdcfe32e60d53ef7b212fe5d56e77376c7e2c59efa512c"]
    }
  }
};

