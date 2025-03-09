const hre = require("hardhat");

async function main() {
  // Get the contract factory
  const AITradingBot = await hre.ethers.getContractFactory("AITradingBot");

  // The QuickSwap Router on Polygon (Mumbai)
  // If deploying to mainnet or another chain, replace this address
  const routerAddress = "0xa5E0829Ca887Ff7F3B5B94b65dBC5545dd37fA4F"; 

  console.log("Deploying AITradingBot contract...");
  const tradingBot = await AITradingBot.deploy(routerAddress);
  await tradingBot.deployed();

  console.log("AITradingBot deployed to:", tradingBot.address);
}

// We recommend this pattern to be able to use async/await everywhere
// and properly handle errors.
main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
