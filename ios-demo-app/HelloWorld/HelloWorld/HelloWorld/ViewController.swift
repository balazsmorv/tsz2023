import UIKit
import CoreML

class ViewController: UIViewController {
    @IBOutlet var imageView: UIImageView!
    @IBOutlet var resultView: UITextView!
    private lazy var module: TorchModule = {
        if let filePath = Bundle.main.path(forResource: "model", ofType: "pt"),
            let module = TorchModule(fileAtPath: filePath) {
            return module
        } else {
            fatalError("Can't find the model file!")
        }
    }()

    private lazy var labels: [String] = {
        if let filePath = Bundle.main.path(forResource: "words", ofType: "txt"),
            let labels = try? String(contentsOfFile: filePath) {
            return labels.components(separatedBy: .newlines)
        } else {
            fatalError("Can't find the text file!")
        }
    }()
    
    let mlmodel = try! mobilenet_v2_torchvision(configuration: MLModelConfiguration())

    override func viewDidLoad() {
        super.viewDidLoad()
        let image = UIImage(named: "image.png")!
        imageView.image = image
        let resizedImage = image.resized(to: CGSize(width: 224, height: 224))
        guard var pixelBuffer = resizedImage.normalized() else {
            return
        }
        
        let clock = ContinuousClock()
        var outputs: [NSNumber]!
        let result = clock.measure {
            outputs = module.predict(image: UnsafeMutableRawPointer(&pixelBuffer))
        }

        print(result)
        
        var out2: mobilenet_v2_torchvisionOutput?
        let result2 = clock.measure {
            out2 = try? mlmodel.prediction(input: mobilenet_v2_torchvisionInput(input_1: image.convertImage()!))
        }
        //print(out2?.output_1.sorted(by: { (one, two) in
        //    one.value > two.value
        //}))
        print(result2)
        
        
        let zippedResults = zip(labels.indices, outputs)
        let sortedResults = zippedResults.sorted { $0.1.floatValue > $1.1.floatValue }.prefix(3)
        var text = ""
        for result in sortedResults {
            text += "\u{2022} \(labels[result.0]) \n\n"
        }
        resultView.text = text
    }
}
