use std::path::PathBuf;

use clap::Parser;

#[derive(Debug, Parser)]
#[command(
    name = "membench-product-conformance",
    about = "Verify the pinned Symbiotic Memory multimodal recall contract"
)]
struct Args {
    /// Local Symbiotic Memory git checkout containing the product-owned contract artifact.
    #[arg(long)]
    product_root: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    membench::multimodal::verify_pinned_product_contract(&args.product_root)?;
    println!(
        "verified Symbiotic Memory multimodal contract {} at {}",
        membench::multimodal::PINNED_PRODUCT_CONTRACT_SHA256,
        membench::multimodal::PINNED_PRODUCT_GIT_SHA
    );
    Ok(())
}
